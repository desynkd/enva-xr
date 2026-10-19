/**
 * Optional password gate for the hosted site (Vercel Routing Middleware, see DEPLOY.md).
 *
 * Off unless the DEMO_PASSWORD environment variable is set in the Vercel project. When it is set,
 * every request needs HTTP Basic auth with that password (any user name, or DEMO_USER if set).
 * This is the free stand-in for Vercel's paid Password Protection.
 */

import crypto from "node:crypto";

export const config = {
	runtime: "nodejs"
};

// Same response as next() from @vercel/functions: let the request through to the static files
function next()
{
	return new Response(null, {headers: {"x-middleware-next": "1"}});
}

function same(a, b)
{
	const ha = crypto.createHash("sha256").update(a).digest();
	const hb = crypto.createHash("sha256").update(b).digest();
	return crypto.timingSafeEqual(ha, hb);
}

export default function middleware(request)
{
	const password = process.env.DEMO_PASSWORD;
	if (!password)
	{
		return next();
	}

	const header = request.headers.get("authorization") || "";
	if (header.startsWith("Basic "))
	{
		const decoded = Buffer.from(header.slice(6), "base64").toString("utf8");
		const colon = decoded.indexOf(":");
		const user = decoded.slice(0, colon);
		const given = decoded.slice(colon + 1);
		const userOk = !process.env.DEMO_USER || same(user, process.env.DEMO_USER);
		if (colon >= 0 && same(given, password) && userOk)
		{
			return next();
		}
	}

	return new Response("Password required.", {
		status: 401,
		headers: {
			"WWW-Authenticate": "Basic realm=\"WebAR experiments\", charset=\"UTF-8\"",
			"Cache-Control": "no-store"
		}
	});
}
