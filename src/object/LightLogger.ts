import {Group, Quaternion, Vector3} from "three";
import {ARRenderer} from "../ARRenderer.ts";
import {ARObject} from "./ARObject.ts";

/**
 * Columns stored for every frame. Missing values are stored as NaN and exported as null.
 *
 * - dir_* is the raw primaryLightDirection, expressed in the light probe space.
 * - probe_* is the pose of the probe space and viewer_* the pose of the device, both in the reference space.
 */
const COLUMNS = [
	"t", "dt", "has_estimate",
	"dir_x", "dir_y", "dir_z",
	"int_r", "int_g", "int_b",
	...Array.from({length: 27}, (v, i) => { return "sh_" + i; }),
	"probe_px", "probe_py", "probe_pz", "probe_qx", "probe_qy", "probe_qz", "probe_qw",
	"viewer_px", "viewer_py", "viewer_pz", "viewer_qx", "viewer_qy", "viewer_qz", "viewer_qw",
	"emulated", "changed", "fallback", "marker"
];

const STRIDE = COLUMNS.length;

/**
 * Lamp spots that can be marked in one capture.
 */
export type LampSpot = "A" | "B";
const C: {[name: string]: number} = {};
COLUMNS.forEach((name, i) => { C[name] = i; });

/**
 * Light logger records the light estimate reported by WebXR every frame, together with the poses needed to express it in world space.
 *
 * Rows are kept in a preallocated typed array and only serialized when the capture is downloaded, to keep the per-frame cost low.
 */
export class LightLogger extends Group implements ARObject
{
	public isARObject = true;

	/**
	 * Tolerance (degrees) used to flag a direction as the (0, 1, 0) fallback.
	 */
	public fallbackToleranceDeg = 0.5;

	/**
	 * Label of the capture, used as file name.
	 */
	public label = "capture";

	/**
	 * Whether frames are being recorded.
	 */
	public recording = false;

	/**
	 * Frames seen since the logger was created, recorded or not.
	 */
	public frameCount = 0;

	/**
	 * Frames recorded in the current capture.
	 */
	public rowCount = 0;

	/**
	 * Whether the current capture was already downloaded.
	 */
	public saved = true;

	/**
	 * Floor point in the reference space.
	 */
	public floor: number[] = null;

	/**
	 * Lamp marks in the reference space, per lamp spot.
	 *
	 * A is where the lamp stands when recording starts, B is where it is slid to in lamp-move runs.
	 */
	public lamps: {[spot in LampSpot]: number[][]} = {A: [], B: []};

	/**
	 * Time markers added during the capture.
	 */
	public markers: {name: string, t: number, row: number}[] = [];

	/**
	 * Latest values, used for the on-screen display.
	 */
	public latestDirection: number[] = null;

	public latestWorldDirection: Vector3 = null;

	public latestViewerPosition: number[] = null;

	public latestEmulated = false;

	public latestChanged = false;

	public latestFallback = false;

	public updates = 0;

	private data = new Float64Array(STRIDE * 18000);

	private lastTime = NaN;

	private lastDirection: number[] = null;

	private pendingMarker = 0;

	private startedAt: string = null;

	private device: any = {userAgent: navigator.userAgent};

	private probeOrientation = new Quaternion();

	public constructor()
	{
		super();

		// @ts-ignore
		const uaData = navigator.userAgentData;
		if (uaData && uaData.getHighEntropyValues)
		{
			uaData.getHighEntropyValues(["model", "platformVersion", "fullVersionList"]).then((values: any) =>
			{
				Object.assign(this.device, values);
			});
		}
	}

	public start(): void
	{
		this.rowCount = 0;
		this.markers = [];
		this.pendingMarker = 0;
		this.startedAt = new Date().toISOString();
		this.recording = true;
		this.saved = false;
	}

	public stop(): void
	{
		this.recording = false;
	}

	public markFloor(position: Vector3): void
	{
		this.floor = position.toArray();
	}

	/**
	 * Mark a lamp spot using the current device position (camera held at the mouth of the lamp).
	 */
	public markLamp(renderer: ARRenderer, spot: LampSpot = "A"): boolean
	{
		if (!renderer.xrViewerPose)
		{
			return false;
		}

		const p = renderer.xrViewerPose.transform.position;
		this.lamps[spot].push([p.x, p.y, p.z]);
		return true;
	}

	/**
	 * Add a time marker, attached to the next recorded frame.
	 */
	public mark(name: string): void
	{
		this.markers.push({name: name, t: this.lastTime, row: this.rowCount});
		this.pendingMarker = this.markers.length;
	}

	/**
	 * Average of the marks of one lamp spot.
	 */
	public lampPosition(spot: LampSpot = "A"): Vector3
	{
		const marks = this.lamps[spot];
		if (marks.length === 0)
		{
			return null;
		}

		const mean = new Vector3();
		for (const l of marks)
		{
			mean.add(new Vector3().fromArray(l));
		}
		return mean.divideScalar(marks.length);
	}

	/**
	 * Largest distance (m) from a mark of one lamp spot to their average.
	 */
	public lampSpread(spot: LampSpot = "A"): number
	{
		const lamp = this.lampPosition(spot);
		if (!lamp)
		{
			return null;
		}

		return Math.max(...this.lamps[spot].map((l) => { return lamp.distanceTo(new Vector3().fromArray(l)); }));
	}

	/**
	 * Ground truth light direction for one lamp spot, normalize(lamp - floor).
	 */
	public trueDirection(spot: LampSpot = "A"): Vector3
	{
		const lamp = this.lampPosition(spot);
		if (!lamp || !this.floor)
		{
			return null;
		}

		return lamp.sub(new Vector3().fromArray(this.floor)).normalize();
	}

	public beforeARUpdate(renderer: ARRenderer, time: number, frame: XRFrame): void
	{
		this.frameCount++;

		const dt = time - this.lastTime;
		this.lastTime = time;

		const probe = renderer.xrLightProbe;
		// @ts-ignore
		const estimate = probe ? frame.getLightEstimate(probe) : null;
		const probePose = probe ? frame.getPose(probe.probeSpace, renderer.xrReferenceSpace) : null;
		const viewer = renderer.xrViewerPose;

		let changed = false;
		let fallback = false;
		if (estimate)
		{
			const d = estimate.primaryLightDirection;
			const dir = [d.x, d.y, d.z];
			const last = this.lastDirection;
			changed = !last || last[0] !== dir[0] || last[1] !== dir[1] || last[2] !== dir[2];
			fallback = d.y / Math.hypot(d.x, d.y, d.z) >= Math.cos(this.fallbackToleranceDeg * Math.PI / 180);

			if (changed)
			{
				this.updates++;
			}
			this.lastDirection = dir;
			this.latestDirection = dir;
			this.latestWorldDirection = null;
			if (probePose)
			{
				const o = probePose.transform.orientation;
				this.probeOrientation.set(o.x, o.y, o.z, o.w);
				this.latestWorldDirection = new Vector3(d.x, d.y, d.z).applyQuaternion(this.probeOrientation);
			}
		}
		this.latestChanged = changed;
		this.latestFallback = fallback;
		this.latestViewerPosition = viewer ? [viewer.transform.position.x, viewer.transform.position.y, viewer.transform.position.z] : null;
		this.latestEmulated = viewer ? viewer.emulatedPosition : false;

		if (!this.recording)
		{
			return;
		}

		if ((this.rowCount + 1) * STRIDE > this.data.length)
		{
			const grown = new Float64Array(this.data.length * 2);
			grown.set(this.data);
			this.data = grown;
		}

		const row = this.data.subarray(this.rowCount * STRIDE, (this.rowCount + 1) * STRIDE);
		row.fill(NaN);
		row[C.t] = time;
		row[C.dt] = this.rowCount > 0 ? dt : NaN;
		row[C.has_estimate] = estimate ? 1 : 0;
		if (estimate)
		{
			const d = estimate.primaryLightDirection;
			const i = estimate.primaryLightIntensity;
			row.set([d.x, d.y, d.z, i.x, i.y, i.z], C.dir_x);
			row.set(estimate.sphericalHarmonicsCoefficients, C.sh_0);
			row[C.changed] = changed ? 1 : 0;
			row[C.fallback] = fallback ? 1 : 0;
		}
		if (probePose)
		{
			const p = probePose.transform.position, o = probePose.transform.orientation;
			row.set([p.x, p.y, p.z, o.x, o.y, o.z, o.w], C.probe_px);
		}
		if (viewer)
		{
			const p = viewer.transform.position, o = viewer.transform.orientation;
			row.set([p.x, p.y, p.z, o.x, o.y, o.z, o.w], C.viewer_px);
			row[C.emulated] = viewer.emulatedPosition ? 1 : 0;
		}
		row[C.marker] = this.pendingMarker;
		this.pendingMarker = 0;

		this.rowCount++;
	}

	public toCapture(): any
	{
		const perSpot = (f: (spot: LampSpot) => any) => { return {A: f("A"), B: f("B")}; };
		const rows: number[][] = [];
		for (let r = 0; r < this.rowCount; r++)
		{
			rows.push(Array.from(this.data.subarray(r * STRIDE, (r + 1) * STRIDE)));
		}

		return {
			label: this.label,
			startedAt: this.startedAt,
			savedAt: new Date().toISOString(),
			device: this.device,
			fallbackToleranceDeg: this.fallbackToleranceDeg,
			floor: this.floor,
			lamps: this.lamps,
			lampMean: perSpot((spot) => { const l = this.lampPosition(spot); return l ? l.toArray() : null; }),
			lampSpread: perSpot((spot) => { return this.lampSpread(spot); }),
			trueDirection: perSpot((spot) => { const d = this.trueDirection(spot); return d ? d.toArray() : null; }),
			markers: this.markers,
			columns: COLUMNS,
			rows: rows
		};
	}

	/**
	 * Download the capture as a JSON file.
	 */
	public download(): void
	{
		const blob = new Blob([JSON.stringify(this.toCapture())], {type: "application/json"});
		const a = document.createElement("a");
		a.href = URL.createObjectURL(blob);
		a.download = this.label + ".json";
		a.click();
		setTimeout(() => { URL.revokeObjectURL(a.href); }, 10000);
		this.saved = true;
	}
}
