/**
 * Spike 2 worker: runs a network on the small camera picture, off the main thread.
 *
 * This file is served as-is from assets, outside the bundle, so it can load onnxruntime itself and
 * keep its own wasm files next to it.
 *
 * Messages in:
 *   {type: "load", model, backend, size}   load a network and warm it up
 *   {type: "run", pixels, sentAt, t, frame, pose, aspect}   one picture, RGBA bytes, transferred
 *
 * Messages out:
 *   {type: "loaded", threads, isolated, warmup}
 *   {type: "result", inferMs, prepMs, direction, peak, ...}
 *   {type: "error", message}
 */

/* global ort, importScripts */

// The webgpu build also carries the plain wasm backend, so both are measured on the same binary
importScripts("ort/ort.webgpu.min.js");

const WARMUP = 5;

// Pictures are RGBA bytes from the camera grab; the stand-in networks want planar RGB, scaled the usual ImageNet way
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// The one trained network: the EMLight regression network (Zhan et al., AAAI 2021), exported by
// emlight-experiment/pytorch/export_onnx.py. It wants a 4:3 picture scaled its own way, see prepareTrained.
const TRAINED = "emlight";
const TRAINED_WIDTH = 256;
const TRAINED_HEIGHT = 192;

let session = null;
let inputName = null;
let size = 256;
let input = null;
let trained = false;
let inWidth = 256;
let inHeight = 256;
let resampled = null;
const histogram = new Uint32Array(256);

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
	trained = message.model === TRAINED;
	inWidth = trained ? TRAINED_WIDTH : size;
	inHeight = trained ? TRAINED_HEIGHT : size;
	input = new Float32Array(3 * inWidth * inHeight);
	resampled = new Float32Array(3 * inWidth * inHeight);

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
		prepare(blank, 1);
		await session.run(feed());
		warmup.push(performance.now() - t0);
	}

	self.postMessage({type: "loaded", threads: ort.env.wasm.numThreads, isolated: self.crossOriginIsolated, warmup: warmup});
}

/**
 * Prepare a picture for the trained network, the same way emlight-experiment/pytorch/infer.py does.
 *
 * The camera grab squashes the whole camera image into a square. The network was trained on 4:3 landscape
 * crops, so the middle 4:3 part of the camera image is cut out and stretched back to its true shape.
 * Then the values are scaled so that their median becomes 0.5, as the training pictures were.
 *
 * aspect is the camera image width over height.
 */
function prepareTrained(pixels, aspect)
{
	const target = inWidth / inHeight;
	const cropW = aspect > target ? size * target / aspect : size;
	const cropH = aspect > target ? size : size * aspect / target;
	const x0 = (size - cropW) / 2;
	const y0 = (size - cropH) / 2;
	const area = inWidth * inHeight;

	histogram.fill(0);
	let count = 0;

	for (let v = 0, i = 0; v < inHeight; v++)
	{
		const sy = Math.min(size - 1, Math.max(0, y0 + (v + 0.5) * cropH / inHeight - 0.5));
		const ya = Math.floor(sy), yb = Math.min(size - 1, ya + 1), fy = sy - ya;

		for (let u = 0; u < inWidth; u++, i++)
		{
			const sx = Math.min(size - 1, Math.max(0, x0 + (u + 0.5) * cropW / inWidth - 0.5));
			const xa = Math.floor(sx), xb = Math.min(size - 1, xa + 1), fx = sx - xa;
			const aa = (ya * size + xa) * 4, ab = (ya * size + xb) * 4, ba = (yb * size + xa) * 4, bb = (yb * size + xb) * 4;

			for (let c = 0; c < 3; c++)
			{
				const top = pixels[aa + c] * (1 - fx) + pixels[ab + c] * fx;
				const bottom = pixels[ba + c] * (1 - fx) + pixels[bb + c] * fx;
				const value = top * (1 - fy) + bottom * fy;
				resampled[c * area + i] = value;

				const bin = Math.round(value);
				if (bin > 0)
				{
					histogram[bin]++;
					count++;
				}
			}
		}
	}

	// Median of the non-zero values, read off the histogram
	let median = 0;
	for (let bin = 1, seen = 0; bin < 256; bin++)
	{
		seen += histogram[bin];
		if (seen * 2 >= count)
		{
			median = bin;
			break;
		}
	}

	const scale = median > 0 ? 0.5 / median : 0;
	for (let i = 0; i < 3 * area; i++)
	{
		input[i] = Math.min(1, resampled[i] * scale);
	}
}

function prepare(pixels, aspect)
{
	if (trained)
	{
		prepareTrained(pixels, aspect);
		return;
	}

	const area = size * size;
	for (let i = 0, p = 0; i < area; i++, p += 4)
	{
		input[i] = (pixels[p] / 255 - MEAN[0]) / STD[0];
		input[area + i] = (pixels[p + 1] / 255 - MEAN[1]) / STD[1];
		input[2 * area + i] = (pixels[p + 2] / 255 - MEAN[2]) / STD[2];
	}
}

function feed()
{
	const feeds = {};
	feeds[inputName] = new ort.Tensor("float32", input, [1, 3, inHeight, inWidth]);
	return feeds;
}

/**
 * How much of the positive anchor weight sits on the strongest anchor. Near 1 means one clear light.
 * Only the trained network has anchor weights; for the stand-ins this is NaN.
 */
function peakShare(distribution)
{
	if (!distribution)
	{
		return NaN;
	}

	let sum = 0, max = 0;
	for (const w of distribution.data)
	{
		if (w > 0)
		{
			sum += w;
			max = Math.max(max, w);
		}
	}

	return sum > 0 ? max / sum : 0;
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
			prepare(pixels, message.aspect || 1);
			const prepMs = performance.now() - t0;

			const t1 = performance.now();
			const out = await running.run(feed());
			const inferMs = performance.now() - t1;

			if (running !== session)
			{
				// Another network was loaded while this ran, so the answer belongs to nothing.
				// The page is still told, otherwise it waits for a reply that never comes.
				self.postMessage({type: "stale", sentAt: message.sentAt});
				return;
			}

			const direction = Array.from(out[running.outputNames[0]].data).slice(0, 3);

			self.postMessage({
				type: "result",
				prepMs: prepMs,
				inferMs: inferMs,
				direction: direction,
				peak: peakShare(out.distribution),
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
