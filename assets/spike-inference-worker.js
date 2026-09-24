/**
 * Spike 2 worker: runs a network on the small camera picture, off the main thread.
 *
 * This file is served as-is from assets, outside the bundle, so it can load onnxruntime itself and
 * keep its own wasm files next to it.
 *
 * Messages in:
 *   {type: "load", model, backend, size}   load a network and warm it up
 *   {type: "run", pixels, sentAt, t, frame, pose}   one picture, RGBA bytes, transferred
 *
 * Messages out:
 *   {type: "loaded", threads, isolated, warmup}
 *   {type: "result", inferMs, prepMs, direction, ...}
 *   {type: "error", message}
 */

/* global ort, importScripts */

// The webgpu build also carries the plain wasm backend, so both are measured on the same binary
importScripts("ort/ort.webgpu.min.js");

const WARMUP = 5;

// Pictures are RGBA bytes from the camera grab; networks want planar RGB, scaled the usual ImageNet way
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;
let inputName = null;
let size = 256;
let input = null;

function threadCount()
{
	if (!self.crossOriginIsolated)
	{
		// Without the COOP/COEP headers the browser only allows one thread
		return 1;
	}

	return Math.min(4, Math.max(1, navigator.hardwareConcurrency || 1));
}

async function load(message)
{
	session = null;
	size = message.size || 256;
	input = new Float32Array(3 * size * size);

	ort.env.wasm.wasmPaths = new URL("ort/", self.location.href).href;
	ort.env.wasm.numThreads = threadCount();
	ort.env.wasm.simd = true;

	if (message.backend === "webgpu" && !navigator.gpu)
	{
		throw new Error("WebGPU is not available in this worker");
	}

	session = await ort.InferenceSession.create("models/" + message.model + ".onnx", {
		executionProviders: [message.backend],
		graphOptimizationLevel: "all"
	});
	inputName = session.inputNames[0];

	// The first runs compile shaders and warm caches, so they are reported separately and never counted
	const warmup = [];
	const blank = new Uint8ClampedArray(size * size * 4);
	for (let i = 0; i < WARMUP; i++)
	{
		const t0 = performance.now();
		await infer(blank);
		warmup.push(performance.now() - t0);
	}

	self.postMessage({type: "loaded", threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated, warmup: warmup});
}

function prepare(pixels)
{
	const area = size * size;
	for (let i = 0, p = 0; i < area; i++, p += 4)
	{
		input[i] = (pixels[p] / 255 - MEAN[0]) / STD[0];
		input[area + i] = (pixels[p + 1] / 255 - MEAN[1]) / STD[1];
		input[2 * area + i] = (pixels[p + 2] / 255 - MEAN[2]) / STD[2];
	}
}

async function infer(pixels)
{
	prepare(pixels);
	const feeds = {};
	feeds[inputName] = new ort.Tensor("float32", input, [1, 3, size, size]);
	const out = await session.run(feeds);
	return out[session.outputNames[0]].data;
}

self.onmessage = async (e) =>
{
	const message = e.data;

	try
	{
		if (message.type === "load")
		{
			await load(message);
			return;
		}

		if (message.type === "run")
		{
			// Held locally: loading another network replaces session while this run is still in flight
			const running = session;
			if (!running)
			{
				return;
			}

			const pixels = new Uint8ClampedArray(message.pixels);

			const t0 = performance.now();
			prepare(pixels);
			const prepMs = performance.now() - t0;

			const t1 = performance.now();
			const feeds = {};
			feeds[inputName] = new ort.Tensor("float32", input, [1, 3, size, size]);
			const out = await running.run(feeds);
			const inferMs = performance.now() - t1;

			if (running !== session)
			{
				// Another network was loaded while this ran, so the answer belongs to nothing
				return;
			}

			const direction = Array.from(out[running.outputNames[0]].data).slice(0, 3);

			self.postMessage({
				type: "result",
				prepMs: prepMs,
				inferMs: inferMs,
				direction: direction,
				sentAt: message.sentAt,
				t: message.t,
				frame: message.frame,
				pose: message.pose
			});
		}
	}
	catch (error)
	{
		self.postMessage({type: "error", message: error && error.message ? error.message : String(error)});
	}
};
