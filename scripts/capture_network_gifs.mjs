#!/usr/bin/env node
/**
 * Capture network (transmission lines only) GIF for each region + all-Japan.
 *
 * Usage:
 *   npx playwright install chromium   # first time only
 *   node scripts/capture_network_gifs.mjs
 *
 * Output:
 *   docs/assets/gif/network_all.gif
 *   docs/assets/gif/network_hokkaido.gif
 *   docs/assets/gif/network_tohoku.gif
 *   ...
 *
 * Requires: playwright, ffmpeg
 */

import { chromium } from "playwright";
import { execSync } from "child_process";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");
const OUT_DIR = path.join(ROOT, "docs", "assets", "gif");
const FRAME_DIR = path.join(ROOT, ".tmp_frames");
const SITE_URL = "https://lutelute.github.io/All-Japan-Grid/";

// GIF settings
const WIDTH = 800;
const HEIGHT = 600;
const FADE_FRAMES = 12;      // frames for zoom-in animation
const HOLD_FRAMES = 18;      // frames to hold on the final view
const FRAME_DELAY_MS = 80;   // ~12.5 fps
const GIF_FPS = 12;

const REGIONS = [
    { id: null,        label: "all",       name: "All Japan" },
    { id: "hokkaido",  label: "hokkaido",  name: "Hokkaido" },
    { id: "tohoku",    label: "tohoku",    name: "Tohoku" },
    { id: "tokyo",     label: "tokyo",     name: "Tokyo" },
    { id: "chubu",     label: "chubu",     name: "Chubu" },
    { id: "hokuriku",  label: "hokuriku",  name: "Hokuriku" },
    { id: "kansai",    label: "kansai",    name: "Kansai" },
    { id: "chugoku",   label: "chugoku",   name: "Chugoku" },
    { id: "shikoku",   label: "shikoku",   name: "Shikoku" },
    { id: "kyushu",    label: "kyushu",    name: "Kyushu" },
    { id: "okinawa",   label: "okinawa",   name: "Okinawa" },
];

function ensureDir(dir) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function cleanDir(dir) {
    if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });
    fs.mkdirSync(dir, { recursive: true });
}

function framesToGif(frameDir, outPath) {
    // Use ffmpeg: input PNG sequence → palette → GIF
    const palette = path.join(frameDir, "palette.png");
    execSync(
        `ffmpeg -y -framerate ${GIF_FPS} -i "${frameDir}/frame_%04d.png" ` +
        `-vf "palettegen=max_colors=128:stats_mode=diff" "${palette}"`,
        { stdio: "pipe" }
    );
    execSync(
        `ffmpeg -y -framerate ${GIF_FPS} -i "${frameDir}/frame_%04d.png" ` +
        `-i "${palette}" -lavfi "paletteuse=dither=bayer:bayer_scale=3" ` +
        `-loop 0 "${outPath}"`,
        { stdio: "pipe" }
    );
    console.log(`  -> ${path.relative(ROOT, outPath)}`);
}

async function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

async function captureRegionGif(page, region, outPath) {
    cleanDir(FRAME_DIR);
    let frameNum = 0;

    const pad = (n) => String(n).padStart(4, "0");

    // If specific region: start from all-Japan, then zoom in
    if (region.id) {
        // Reset to all-Japan view first
        await page.evaluate(() => {
            if (typeof selectRegion === "function") selectRegion(null);
        });
        await sleep(1500);

        // Capture a few frames of the all-Japan view
        for (let i = 0; i < 4; i++) {
            await page.screenshot({
                path: path.join(FRAME_DIR, `frame_${pad(frameNum++)}.png`),
            });
            await sleep(FRAME_DELAY_MS);
        }

        // Click region to trigger zoom
        await page.evaluate((rid) => {
            if (typeof selectRegion === "function") selectRegion(rid);
        }, region.id);

        // Capture frames during zoom animation
        for (let i = 0; i < FADE_FRAMES; i++) {
            await sleep(FRAME_DELAY_MS);
            await page.screenshot({
                path: path.join(FRAME_DIR, `frame_${pad(frameNum++)}.png`),
            });
        }
    }

    // Wait for tiles to load
    await sleep(2000);

    // Capture hold frames (final view)
    for (let i = 0; i < HOLD_FRAMES; i++) {
        await page.screenshot({
            path: path.join(FRAME_DIR, `frame_${pad(frameNum++)}.png`),
        });
        await sleep(FRAME_DELAY_MS);
    }

    // Build GIF
    framesToGif(FRAME_DIR, outPath);
}

async function main() {
    ensureDir(OUT_DIR);

    console.log("Launching browser...");
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        viewport: { width: WIDTH, height: HEIGHT },
        deviceScaleFactor: 2,
    });
    const page = await context.newPage();

    console.log(`Loading ${SITE_URL}`);
    await page.goto(SITE_URL, { waitUntil: "networkidle" });
    await sleep(3000); // wait for GeoJSON to load

    // Switch to エリア tab (loads all voltages, region coloring)
    console.log("Switching to area tab...");
    await page.click('[data-tab="tab-area"]');
    await sleep(2000);

    // Hide substations and plants, keep only lines
    console.log("Hiding substations & plants...");
    await page.evaluate(() => {
        document.querySelectorAll('.layer-cb[data-layer="subs"]').forEach((cb) => {
            if (cb.checked) { cb.checked = false; cb.dispatchEvent(new Event("change")); }
        });
        document.querySelectorAll('.layer-cb[data-layer="plants"]').forEach((cb) => {
            if (cb.checked) { cb.checked = false; cb.dispatchEvent(new Event("change")); }
        });
    });
    await sleep(1500);

    // Also hide sidebar for cleaner capture
    await page.evaluate(() => {
        var sidebar = document.getElementById("sidebar");
        if (sidebar) sidebar.style.display = "none";
        var mapEl = document.getElementById("map");
        if (mapEl) mapEl.style.flex = "1";
        // Trigger Leaflet resize
        if (typeof map !== "undefined" && map.invalidateSize) {
            map.invalidateSize();
        }
    });
    await sleep(1000);

    // Capture each region
    for (const region of REGIONS) {
        console.log(`Capturing: ${region.name}...`);
        const outFile = path.join(OUT_DIR, `network_${region.label}.gif`);
        await captureRegionGif(page, region, outFile);
    }

    // Cleanup
    await browser.close();
    if (fs.existsSync(FRAME_DIR)) fs.rmSync(FRAME_DIR, { recursive: true });

    console.log(`\nDone! ${REGIONS.length} GIFs saved to docs/assets/gif/`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
