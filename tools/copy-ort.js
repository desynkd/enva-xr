/**
 * Copies the onnxruntime-web files the spike 2 worker needs into assets/ort.
 *
 *     node tools/copy-ort.js
 *
 * The bundle loads its WebAssembly glue with a dynamic import, so the file names have to match the
 * ones the bundle asks for. They changed between onnxruntime versions (the webgpu bundle wants the
 * asyncify build, not the jsep one), so the list is read out of the bundle itself rather than guessed.
 */

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const from = path.join(root, "node_modules", "onnxruntime-web", "dist");
const to = path.join(root, "assets", "ort");

// The webgpu bundle also carries the plain wasm backend, so both are measured on the same binary
const BUNDLE = "ort.webgpu.min.js";

const bundle = fs.readFileSync(path.join(from, BUNDLE), "utf8");
const glue = [...new Set(bundle.match(/ort-wasm[a-z0-9.-]*\.mjs/g) || [])];
if (glue.length === 0)
{
	console.error("Could not find the WebAssembly glue file name inside " + BUNDLE);
	process.exit(1);
}

// Each glue file loads its own .wasm, named inside it
const files = [BUNDLE];
for (const mjs of glue)
{
	files.push(mjs);
	const wasm = [...new Set(fs.readFileSync(path.join(from, mjs), "utf8").match(/ort-wasm[a-z0-9.-]*\.wasm/g) || [])];
	files.push(...wasm);
}

fs.mkdirSync(to, {recursive: true});
for (const name of fs.readdirSync(to))
{
	fs.rmSync(path.join(to, name));
}

for (const name of files)
{
	fs.copyFileSync(path.join(from, name), path.join(to, name));
	console.log(`${name}  ${(fs.statSync(path.join(to, name)).size / 1e6).toFixed(1)} MB`);
}

console.log("\nCopied into assets/ort. Restart the dev server so the files are served.");
