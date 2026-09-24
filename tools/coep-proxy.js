/**
 * Adds the two cross-origin isolation headers in front of the Parcel dev server, which cannot send them itself.
 *
 * WebAssembly only runs on several cores when the page is cross-origin isolated, which needs both
 * Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy (spike 2, assumption 4).
 *
 *     npm start                             # Parcel on 9000 (https)
 *     node tools/coep-proxy.js              # this, on 9001 (https), isolation ON
 *     node tools/coep-proxy.js --no-isolate # same, isolation OFF, to compare one core against several
 *     ngrok http https://localhost:9001
 *
 * The certificate is self-signed, so the phone shows the same warning as before.
 */

import fs from "node:fs";
import https from "node:https";
import tls from "node:tls";
import path from "node:path";
import {fileURLToPath} from "node:url";

const PORT = Number(process.env.PORT || 9001);
const UPSTREAM = Number(process.env.UPSTREAM || 9000);
const isolate = !process.argv.includes("--no-isolate");

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

// Parcel writes the certificate it generates for --https into .parcel-cache, otherwise fall back to the project one
const CERTS = [
	{key: path.join(root, ".parcel-cache", "private.pem"), cert: path.join(root, ".parcel-cache", "primary.crt")},
	{key: path.join(root, "key.pem"), cert: path.join(root, "cert.pem")}
];

const found = CERTS.find((c) => { return fs.existsSync(c.key) && fs.existsSync(c.cert); });
if (!found)
{
	console.error("No certificate found. Start Parcel once with `npm start`, then run this again.");
	process.exit(1);
}

const options = {key: fs.readFileSync(found.key), cert: fs.readFileSync(found.cert)};

function isolationHeaders(headers)
{
	if (!isolate)
	{
		return headers;
	}

	return {
		...headers,
		"cross-origin-opener-policy": "same-origin",
		"cross-origin-embedder-policy": "require-corp",
		// Everything is served from here, so let each file be embedded by the isolated page
		"cross-origin-resource-policy": "same-origin"
	};
}

const server = https.createServer(options, (req, res) =>
{
	const upstream = https.request(
		{host: "localhost", port: UPSTREAM, method: req.method, path: req.url, headers: req.headers, rejectUnauthorized: false},
		(up) =>
		{
			res.writeHead(up.statusCode, isolationHeaders(up.headers));
			up.pipe(res);
		});

	upstream.on("error", (e) =>
	{
		res.writeHead(502, {"content-type": "text/plain"});
		res.end("Parcel is not answering on " + UPSTREAM + ": " + e.message);
	});

	req.pipe(upstream);
});

// Parcel's hot reload uses a websocket, which has to be passed through as a raw socket
server.on("upgrade", (req, socket, head) =>
{
	const up = tls.connect({host: "localhost", port: UPSTREAM, rejectUnauthorized: false}, () =>
	{
		const lines = Object.entries(req.headers).map(([k, v]) => { return k + ": " + v; });
		up.write("GET " + req.url + " HTTP/1.1\r\n" + lines.join("\r\n") + "\r\n\r\n");
		if (head && head.length)
		{
			up.write(head);
		}
		socket.pipe(up);
		up.pipe(socket);
	});

	up.on("error", () => { socket.destroy(); });
	socket.on("error", () => { up.destroy(); });
});

server.listen(PORT, () =>
{
	console.log(`Proxy on https://localhost:${PORT} -> https://localhost:${UPSTREAM}   (certificate: ${path.relative(root, found.cert)})`);
	console.log(`Cross-origin isolation: ${isolate ? "ON (several cores)" : "OFF (one core)"}`);
	console.log(`Point ngrok at it:  ngrok http https://localhost:${PORT}`);
});
