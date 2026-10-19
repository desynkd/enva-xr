/**
 * Builds the example pages into build/ as a static site, for hosting (Vercel, see DEPLOY.md).
 *
 *     node tools/build-site.js
 *
 * Steps, each skipped when the branch does not have the piece it needs:
 *  1. copy the onnxruntime-web files into assets/ort (tools/copy-ort.js)
 *  2. download the FLEA weights into assets/models, checked against a fixed hash
 *  3. parcel build of example/*.html
 *  4. keep only the files the pages use and are allowed to be shared
 *  5. write an index page, the licence notices and a robots.txt
 *
 * vercel.json sends the same cross-origin isolation headers as tools/coep-proxy.js, so the hosted pages
 * get the same thread count as the spike runs.
 *
 * Model weights are never committed. FLEA (CC BY-NC 4.0) is fetched from its author's own release at a
 * pinned revision. EMLight is left out on purpose: its repository has no licence and it was trained on
 * the Laval Indoor dataset, so it is not shared from a hosted site.
 */

import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import {spawnSync} from "node:child_process";
import {fileURLToPath} from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const out = path.join(root, "build");
const assets = path.join(root, "assets");
const worker = path.join(assets, "spike-inference-worker.js");

// FLEA release on Hugging Face, pinned to one revision so the hosted model cannot change underneath a run
const FLEA = {
	url: "https://huggingface.co/spaces/giosanchez0208/fast-lighting-estimation-for-ar/resolve/c915c66d0bbe3b29d13d23a3dd89fd878eb76070/flea.onnx",
	sha256: "839f22ac3cca776dcfd19867c525392f123af6b357e525692a17fc82b7838f1f"
};

const FLEA_NOTICE = `FLEA: Fast Lighting Estimation for AR
(c) 2026 Gio Sanchez
Code:    https://github.com/giosanchez0208/Fast-Lighting-Estimation-For-AR
Weights: https://huggingface.co/spaces/giosanchez0208/fast-lighting-estimation-for-ar (flea.onnx)

Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/).
The weights in flea.onnx are used unmodified, for non-commercial research.
`;

// Model files that may be served. Anything else found in assets/models stays local.
const SHAREABLE_MODELS = ["flea.onnx", "FLEA-NOTICE.txt"];

// Static folders from assets/ that no page loads (upstream glTF samples), so they are not served
const UNUSED_STATIC = ["3d"];

const PAGES = {
	"spike-inference.html": "Inference spike: light-direction networks driving the shadow light (Spikes 2-4)",
	"spike-camera.html": "Camera spike: grab camera frames with pose (Spike 1)",
	"spike-camera-check.html": "Camera-access check (Spike 1)",
	"pilot-logger.html": "Pilot logger: WebXR light probe against a marked lamp",
	"light-probe.html": "WebXR light probe (upstream example)",
	"basic.html": "Basic scene (upstream example)",
	"cursor.html": "Hit-test cursor (upstream example)",
	"depth.html": "Depth occlusion (upstream example)",
	"depth-canvas.html": "Depth canvas (upstream example)",
	"image-target.html": "Image target (upstream example)",
	"multi-feature.html": "Multiple features (upstream example)"
};

function step(message)
{
	console.log("\n== " + message);
}

function sha256(file)
{
	return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function run(args)
{
	const result = spawnSync(process.execPath, args, {cwd: root, stdio: "inherit"});
	if (result.status !== 0)
	{
		process.exit(result.status || 1);
	}
}

async function fetchFlea()
{
	const models = path.join(assets, "models");
	const file = path.join(models, "flea.onnx");
	fs.mkdirSync(models, {recursive: true});

	if (!fs.existsSync(file) || sha256(file) !== FLEA.sha256)
	{
		console.log("Downloading " + FLEA.url);
		const response = await fetch(FLEA.url);
		if (!response.ok)
		{
			console.error("FLEA download failed: " + response.status + " " + response.statusText);
			process.exit(1);
		}
		fs.writeFileSync(file, Buffer.from(await response.arrayBuffer()));
	}

	const hash = sha256(file);
	if (hash !== FLEA.sha256)
	{
		fs.rmSync(file);
		console.error("flea.onnx does not match the pinned hash (got " + hash + "). Not building with it.");
		process.exit(1);
	}

	fs.writeFileSync(path.join(models, "FLEA-NOTICE.txt"), FLEA_NOTICE);
	console.log("flea.onnx ok (" + (fs.statSync(file).size / 1e6).toFixed(1) + " MB)");
}

// Through the API rather than the CLI: package.json declares library targets (types, main, module)
// that the CLI would also apply to the pages. Passing one target here builds only the pages.
async function parcelBuild()
{
	const core = await import("@parcel/core");
	const Parcel = core.default.default || core.default;
	const example = path.join(root, "example");
	const entries = fs.readdirSync(example).filter((name) => name.endsWith(".html")).map((name) => path.join(example, name));

	const parcel = new Parcel({
		entries: entries,
		defaultConfig: "@parcel/config-default",
		mode: "production",
		shouldDisableCache: true,
		targets: {
			site: {distDir: out, publicUrl: "./", sourceMap: false, context: "browser"}
		},
		defaultTargetOptions: {shouldOptimize: true, sourceMaps: false, publicUrl: "./"},
		additionalReporters: [{packageName: "@parcel/reporter-cli", resolveFrom: root}]
	});

	try
	{
		const result = await parcel.run();
		console.log("Built " + result.bundleGraph.getBundles().length + " bundles from " + entries.length + " pages");
	}
	catch (error)
	{
		for (const diagnostic of error.diagnostics || [error])
		{
			console.error(diagnostic.message);
			if (diagnostic.codeFrames)
			{
				console.error(JSON.stringify(diagnostic.codeFrames.map((frame) => frame.filePath)));
			}
		}
		process.exit(1);
	}
}

function prune()
{
	for (const name of UNUSED_STATIC)
	{
		fs.rmSync(path.join(out, name), {recursive: true, force: true});
	}

	const models = path.join(out, "models");
	if (fs.existsSync(models))
	{
		for (const name of fs.readdirSync(models))
		{
			if (!SHAREABLE_MODELS.includes(name))
			{
				fs.rmSync(path.join(models, name), {recursive: true, force: true});
				console.log("Left out models/" + name);
			}
		}
	}
}

function licences()
{
	const parts = [];
	const add = (title, text) => parts.push("=".repeat(78) + "\n" + title + "\n" + "=".repeat(78) + "\n\n" + text.trim() + "\n");

	add("enva-xr (this site's code, a fork of tentone/enva-xr)", fs.readFileSync(path.join(root, "LICENSE.md"), "utf8"));

	for (const name of ["three", "cannon-es"])
	{
		const file = path.join(root, "node_modules", name, "LICENSE");
		if (fs.existsSync(file))
		{
			add(name, fs.readFileSync(file, "utf8"));
		}
	}

	const mit = (holder) => `MIT License

Copyright (c) ${holder}

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.`;

	if (fs.existsSync(path.join(out, "ort")))
	{
		add("onnxruntime-web (files in ort/)", mit("Microsoft Corporation"));
	}
	if (fs.existsSync(path.join(out, "lib", "es-module-shims.js")) || fs.existsSync(path.join(root, "example", "lib", "es-module-shims.js")))
	{
		add("ES Module Shims", mit("2018-2023 Guy Bedford"));
	}
	if (fs.existsSync(path.join(out, "models", "flea.onnx")))
	{
		add("FLEA model weights (models/flea.onnx)", FLEA_NOTICE + "\nNot endorsed by the author. Full licence: https://creativecommons.org/licenses/by-nc/4.0/legalcode");
	}

	fs.writeFileSync(path.join(out, "licenses.txt"), parts.join("\n"));
}

function escape(text)
{
	return text.replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;"})[c]);
}

function index()
{
	const built = fs.readdirSync(out).filter((name) => name.endsWith(".html") && name !== "index.html");
	built.sort((a, b) => (Object.keys(PAGES).indexOf(a) + 1 || 99) - (Object.keys(PAGES).indexOf(b) + 1 || 99));

	const branch = process.env.VERCEL_GIT_COMMIT_REF || "";
	const commit = (process.env.VERCEL_GIT_COMMIT_SHA || "").slice(0, 7);
	const flea = fs.existsSync(path.join(out, "models", "flea.onnx"));

	const items = built.map((name) => `<li><a href="./${name}">${escape(name)}</a><br><span>${escape(PAGES[name] || "")}</span></li>`).join("\n");

	fs.writeFileSync(path.join(out, "index.html"), `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>WebAR light experiments</title>
<style>
	body { font: 16px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 40rem; padding: 1rem; color: #222; background: #fff; }
	li { margin-bottom: 0.8rem; } li span { color: #666; font-size: 0.9rem; }
	footer { margin-top: 2rem; border-top: 1px solid #ddd; padding-top: 1rem; font-size: 0.85rem; color: #555; }
	@media (prefers-color-scheme: dark) { body { color: #ddd; background: #111; } li span, footer { color: #999; } footer { border-color: #333; } a { color: #8ab4f8; } }
</style>
</head>
<body>
<h1>WebAR light experiments</h1>
<p>Final-year research prototype: camera-only light-direction estimation for WebAR shadows. Needs Chrome on an ARCore Android phone.${branch ? `<br>Branch <code>${escape(branch)}</code>${commit ? ` at <code>${escape(commit)}</code>` : ""}.` : ""}</p>
<ul>
${items}
</ul>
<footer>
<p><strong>Privacy.</strong> Camera images are processed on this device only. Nothing is uploaded; recordings are saved to the phone only when you press Download.</p>
${flea ? `<p><strong>FLEA</strong> light-estimation weights &copy; 2026 Gio Sanchez (<a href="https://github.com/giosanchez0208/Fast-Lighting-Estimation-For-AR">code</a>, <a href="https://huggingface.co/spaces/giosanchez0208/fast-lighting-estimation-for-ar">weights</a>), used unmodified under <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a> for non-commercial research. Not endorsed by the author.</p>` : ""}
<p>Built on <a href="https://github.com/tentone/enva-xr">enva-xr</a> (MIT), three.js, cannon-es and onnxruntime-web. <a href="./licenses.txt">Licences</a>.</p>
</footer>
</body>
</html>
`);
}

step("onnxruntime-web files");
if (fs.existsSync(path.join(root, "tools", "copy-ort.js")) && fs.existsSync(path.join(root, "node_modules", "onnxruntime-web")))
{
	run(["tools/copy-ort.js"]);
}
else
{
	console.log("Not used on this branch");
}

step("FLEA weights");
if (fs.existsSync(worker) && /flea/i.test(fs.readFileSync(worker, "utf8")))
{
	await fetchFlea();
}
else
{
	console.log("Not used on this branch");
}

step("Parcel build");
fs.rmSync(out, {recursive: true, force: true});
await parcelBuild();

step("Prune, index, licences");
prune();
licences();
index();
fs.writeFileSync(path.join(out, "robots.txt"), "User-agent: *\nDisallow: /\n");
console.log("Site written to build/");
