import {Group} from "three";
import {ARRenderer} from "../ARRenderer.ts";
import {ARObject} from "./ARObject.ts";

/**
 * How the small picture is copied from the GPU into memory.
 *
 * - off: no grab.
 * - sync: readPixels, which waits for the GPU to finish.
 * - async: readPixels into a pixel buffer, fetched a few frames later once its fence has signaled.
 */
export type CameraGrabMode = "off" | "sync" | "async";

const VERTEX = `#version 300 es
out vec2 vUv;
void main()
{
	// Fullscreen triangle, no vertex buffers needed
	vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
	vUv = p;
	gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT = `#version 300 es
precision mediump float;
uniform sampler2D uCamera;
uniform bool uFlipY;
in vec2 vUv;
out vec4 color;
void main()
{
	vec2 uv = uFlipY ? vec2(vUv.x, 1.0 - vUv.y) : vUv;
	color = vec4(texture(uCamera, uv).rgb, 1.0);
}`;

/**
 * Number of pixel buffers used by the async mode. A new copy is skipped while all of them are waiting on the GPU.
 */
const SLOTS = 3;

/**
 * Number of samples kept for the rolling timing statistics.
 */
const WINDOW = 120;

class Rolling
{
	public values: number[] = [];

	public push(v: number): void
	{
		this.values.push(v);
		if (this.values.length > WINDOW)
		{
			this.values.shift();
		}
	}

	public avg(): number
	{
		return this.values.length ? this.values.reduce((a, b) => { return a + b; }, 0) / this.values.length : NaN;
	}

	public max(): number
	{
		return this.values.length ? Math.max(...this.values) : NaN;
	}
}

interface Slot
{
	pbo: WebGLBuffer;
	sync: WebGLSync;
	frame: number;
	time: number;
	shrinkMs: number;
	readMs: number;
	pose: number[];
}

/**
 * Camera grab takes the camera image of each frame, shrinks it on the GPU and copies the small picture into memory.
 *
 * Requires a WebGL 2 renderer with config.cameraAccess set true.
 *
 * The camera texture is only valid during its own frame, so the shrink runs straight away in beforeARUpdate.
 * The pose of the frame the picture was taken in is kept with it, because the async copy arrives frames later.
 */
export class CameraGrab extends Group implements ARObject
{
	public isARObject = true;

	public mode: CameraGrabMode = "off";

	/**
	 * Grab one frame every N frames.
	 */
	public every = 1;

	/**
	 * Size of the small picture.
	 */
	public width = 256;

	public height = 256;

	/**
	 * If true, row 0 of the pixels is the top of the camera image (assumes the camera texture has a bottom-left origin, as usual in WebGL).
	 */
	public flipY = true;

	/**
	 * Latest small picture, RGBA, row 0 first.
	 */
	public pixels: Uint8Array = null;

	/**
	 * Frame time, frame number and viewer pose [px, py, pz, qx, qy, qz, qw] of the frame the latest picture was taken in.
	 */
	public pixelsTime = NaN;

	public pixelsFrame = 0;

	public pixelsPose: number[] = null;

	/**
	 * Brightness of the latest picture: mean luma (0-255), mean luma of the non-saturated pixels, and the fraction of saturated pixels.
	 */
	public luma = NaN;

	public lumaUnsaturated = NaN;

	public saturated = NaN;

	/**
	 * Saturation threshold, applied to the largest channel.
	 */
	public saturationLevel = 250;

	public cameraWidth = 0;

	public cameraHeight = 0;

	public frameCount = 0;

	public framesWithCamera = 0;

	public grabs = 0;

	public copies = 0;

	/**
	 * Async copies skipped because every pixel buffer was still waiting on the GPU.
	 */
	public skipped = 0;

	/**
	 * Whether the last frame had a viewer pose. Without one there are no views, so no camera image.
	 */
	public hasPose = false;

	/**
	 * Frames where a grab failed but the capture carried on. The last message is kept in error.
	 */
	public errors = 0;

	public error: string = null;

	/**
	 * Main thread time (ms) of each step.
	 *
	 * - getCameraImage: obtaining the camera texture.
	 * - shrink: issuing the draw into the small render target (CPU side only, the GPU work runs later).
	 * - read: readPixels, which in sync mode includes waiting for the GPU to finish the shrink.
	 * - fetch: getBufferSubData of a finished async copy.
	 * - total: all the grab work in a frame, including the three.js state reset.
	 * - latency: frames between issuing an async copy and fetching it.
	 */
	public stats = {
		getCameraImage: new Rolling(),
		shrink: new Rolling(),
		read: new Rolling(),
		fetch: new Rolling(),
		total: new Rolling(),
		latency: new Rolling()
	};

	/**
	 * Whether completed copies are being recorded.
	 */
	public recording = false;

	/**
	 * One row per completed copy while recording.
	 */
	public records: number[][] = [];

	private gl: WebGL2RenderingContext = null;

	private program: WebGLProgram = null;

	private vao: WebGLVertexArrayObject = null;

	private target: WebGLTexture = null;

	private framebuffer: WebGLFramebuffer = null;

	private uCamera: WebGLUniformLocation = null;

	private uFlipY: WebGLUniformLocation = null;

	private slots: Slot[] = [];

	private pending: Slot[] = [];

	public start(): void
	{
		this.records = [];
		this.recording = true;
	}

	public stop(): void
	{
		this.recording = false;
	}

	public beforeARUpdate(renderer: ARRenderer, time: number, frame: XRFrame): void
	{
		this.frameCount++;

		// The views must come from the pose of this frame. renderer.xrViews keeps the views of the last frame that had a
		// pose, and asking those for a camera image throws, because their frame is no longer the active one.
		const pose = renderer.xrViewerPose;
		const view: any = pose && pose.views.length > 0 ? pose.views[0] : null;
		const camera = view ? view.camera : null;
		this.hasPose = pose !== null && pose !== undefined;
		if (camera)
		{
			this.framesWithCamera++;
			this.cameraWidth = camera.width;
			this.cameraHeight = camera.height;
		}

		const grab = this.mode !== "off" && camera && renderer.xrGlBinding && this.frameCount % this.every === 0;
		if (!grab && this.pending.length === 0)
		{
			return;
		}

		// @ts-ignore
		const gl: WebGL2RenderingContext = renderer.glContext;
		if (!(gl instanceof WebGL2RenderingContext))
		{
			this.fail("CameraGrab requires WebGL 2 (config.useWebGL2)");
			return;
		}

		const three = renderer.renderer;
		const renderTarget = three.getRenderTarget();
		const t0 = performance.now();
		let ready = this.gl === gl;

		try
		{
			if (!ready)
			{
				this.setup(gl);
				ready = true;
			}

			this.poll();

			if (grab)
			{
				this.grab(renderer, view, camera, time);
			}
		}
		catch (e)
		{
			const message = e && e.message ? e.message : String(e);
			if (ready)
			{
				// A single bad frame (e.g. tracking dropped) must not end the capture
				this.errors++;
				this.error = message;
			}
			else
			{
				this.fail(message);
			}
		}
		finally
		{
			// three.js caches GL state, so it must forget what was changed here and bind the XR framebuffer again
			three.resetState();
			three.setRenderTarget(renderTarget);
		}

		this.stats.total.push(performance.now() - t0);
	}

	private fail(message: string): void
	{
		this.error = message;
		this.mode = "off";
		console.error("enva-xr: CameraGrab", message);
	}

	private setup(gl: WebGL2RenderingContext): void
	{
		this.gl = gl;

		const compile = (type: number, source: string): WebGLShader =>
		{
			const shader = gl.createShader(type);
			gl.shaderSource(shader, source);
			gl.compileShader(shader);
			if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
			{
				throw new Error("Shader: " + gl.getShaderInfoLog(shader));
			}
			return shader;
		};

		this.program = gl.createProgram();
		gl.attachShader(this.program, compile(gl.VERTEX_SHADER, VERTEX));
		gl.attachShader(this.program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
		gl.linkProgram(this.program);
		if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
		{
			throw new Error("Program: " + gl.getProgramInfoLog(this.program));
		}
		this.uCamera = gl.getUniformLocation(this.program, "uCamera");
		this.uFlipY = gl.getUniformLocation(this.program, "uFlipY");

		// Empty vertex array, so the draw does not touch the attributes of three.js
		this.vao = gl.createVertexArray();

		this.target = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, this.target);
		gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, this.width, this.height);

		this.framebuffer = gl.createFramebuffer();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.target, 0);
		if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
		{
			throw new Error("Small render target is not complete");
		}

		const size = this.width * this.height * 4;
		this.pixels = new Uint8Array(size);
		this.slots = [];
		for (let i = 0; i < SLOTS; i++)
		{
			const pbo = gl.createBuffer();
			gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
			gl.bufferData(gl.PIXEL_PACK_BUFFER, size, gl.STREAM_READ);
			this.slots.push({pbo: pbo, sync: null, frame: 0, time: NaN, shrinkMs: NaN, readMs: NaN, pose: null});
		}
		gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
	}

	/**
	 * Fetch the async copies whose fence has signaled, oldest first.
	 */
	private poll(): void
	{
		const gl = this.gl;

		while (this.pending.length > 0)
		{
			const slot = this.pending[0];
			const status = gl.clientWaitSync(slot.sync, 0, 0);
			if (status !== gl.ALREADY_SIGNALED && status !== gl.CONDITION_SATISFIED)
			{
				return;
			}

			const t0 = performance.now();
			gl.bindBuffer(gl.PIXEL_PACK_BUFFER, slot.pbo);
			gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.pixels);
			gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
			this.stats.fetch.push(performance.now() - t0);
			this.stats.latency.push(this.frameCount - slot.frame);

			gl.deleteSync(slot.sync);
			slot.sync = null;
			this.pending.shift();

			this.completed("async", slot.frame, slot.time, slot.pose, this.frameCount - slot.frame, slot.shrinkMs, slot.readMs);
		}
	}

	private grab(renderer: ARRenderer, view: any, camera: any, time: number): void
	{
		const gl = this.gl;

		// The camera image belongs to this view, so view.transform is the pose it was taken from
		const p = view.transform.position, o = view.transform.orientation;
		const pose = [p.x, p.y, p.z, o.x, o.y, o.z, o.w];

		let slot: Slot = null;
		if (this.mode === "async")
		{
			slot = this.slots.find((s) => { return s.sync === null; });
			if (!slot)
			{
				this.skipped++;
				return;
			}
		}

		let t0 = performance.now();
		// @ts-ignore
		const texture: WebGLTexture = renderer.xrGlBinding.getCameraImage(camera);
		this.stats.getCameraImage.push(performance.now() - t0);
		if (!texture)
		{
			return;
		}

		// Shrink by drawing the camera texture into the small render target
		t0 = performance.now();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
		gl.viewport(0, 0, this.width, this.height);
		gl.disable(gl.SCISSOR_TEST);
		gl.disable(gl.DEPTH_TEST);
		gl.disable(gl.STENCIL_TEST);
		gl.disable(gl.BLEND);
		gl.disable(gl.CULL_FACE);
		gl.colorMask(true, true, true, true);
		gl.useProgram(this.program);
		gl.bindVertexArray(this.vao);
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D, texture);
		gl.uniform1i(this.uCamera, 0);
		gl.uniform1i(this.uFlipY, this.flipY ? 1 : 0);
		gl.drawArrays(gl.TRIANGLES, 0, 3);
		gl.bindVertexArray(null);
		const shrinkMs = performance.now() - t0;
		this.stats.shrink.push(shrinkMs);
		this.grabs++;

		// Copy into memory
		t0 = performance.now();
		gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
		if (this.mode === "sync")
		{
			gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);
			const readMs = performance.now() - t0;
			this.stats.read.push(readMs);
			this.completed("sync", this.frameCount, time, pose, 0, shrinkMs, readMs);
		}
		else
		{
			gl.bindBuffer(gl.PIXEL_PACK_BUFFER, slot.pbo);
			gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, 0);
			gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
			slot.sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
			gl.flush();
			slot.readMs = performance.now() - t0;
			this.stats.read.push(slot.readMs);

			slot.frame = this.frameCount;
			slot.time = time;
			slot.shrinkMs = shrinkMs;
			slot.pose = pose;
			this.pending.push(slot);
		}
	}

	/**
	 * Called when this.pixels holds a new picture.
	 */
	private completed(mode: CameraGrabMode, frame: number, time: number, pose: number[], latency: number, shrinkMs: number, readMs: number): void
	{
		this.copies++;
		this.pixelsFrame = frame;
		this.pixelsTime = time;
		this.pixelsPose = pose;

		const px = this.pixels;
		const level = this.saturationLevel;
		let sum = 0, sumUnsaturated = 0, unsaturated = 0;
		for (let i = 0; i < px.length; i += 4)
		{
			const r = px[i], g = px[i + 1], b = px[i + 2];
			const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
			sum += y;
			if (r < level && g < level && b < level)
			{
				sumUnsaturated += y;
				unsaturated++;
			}
		}
		const count = px.length / 4;
		this.luma = sum / count;
		this.lumaUnsaturated = unsaturated > 0 ? sumUnsaturated / unsaturated : NaN;
		this.saturated = 1 - unsaturated / count;

		if (this.recording)
		{
			this.records.push([
				time, frame, mode === "sync" ? 1 : 2, latency,
				shrinkMs, readMs, mode === "async" ? this.stats.fetch.values[this.stats.fetch.values.length - 1] : NaN,
				this.luma, this.lumaUnsaturated, this.saturated,
				...pose
			]);
		}
	}

	public toCapture(label: string): any
	{
		return {
			label: label,
			savedAt: new Date().toISOString(),
			userAgent: navigator.userAgent,
			width: this.width,
			height: this.height,
			every: this.every,
			flipY: this.flipY,
			cameraWidth: this.cameraWidth,
			cameraHeight: this.cameraHeight,
			saturationLevel: this.saturationLevel,
			columns: ["t", "frame", "mode", "latency_frames", "shrink_ms", "read_ms", "fetch_ms", "luma", "luma_unsat", "saturated", "px", "py", "pz", "qx", "qy", "qz", "qw"],
			modes: {1: "sync", 2: "async"},
			rows: this.records.map((r) => { return r.map((v) => { return Number.isFinite(v) ? v : null; }); })
		};
	}

	public download(label: string): void
	{
		const blob = new Blob([JSON.stringify(this.toCapture(label))], {type: "application/json"});
		const a = document.createElement("a");
		a.href = URL.createObjectURL(blob);
		a.download = label + "-grab.json";
		a.click();
		setTimeout(() => { URL.revokeObjectURL(a.href); }, 10000);
	}
}
