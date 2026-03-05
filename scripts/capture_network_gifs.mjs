#!/usr/bin/env node
/**
 * Capture a single GIF cycling through all regions with region-colored lines.
 *
 * Flow: All Japan → Hokkaido → Tohoku → ... → Okinawa → All Japan
 *
 * Usage:
 *   node scripts/capture_network_gifs.mjs
 *
 * Output:
 *   docs/assets/gif/network_tour.gif
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

const WIDTH = 800;
const HEIGHT = 600;
const GIF_FPS = 12;
const FRAME_DELAY_MS = 80;

const REGIONS = [
    { id: null,        name: "All Japan (open)",  hold: 30 },
    { id: "hokkaido",  name: "Hokkaido",          hold: 18 },
    { id: "tohoku",    name: "Tohoku",            hold: 18 },
    { id: "tokyo",     name: "Tokyo",             hold: 18 },
    { id: "chubu",     name: "Chubu",             hold: 18 },
    { id: "hokuriku",  name: "Hokuriku",          hold: 18 },
    { id: "kansai",    name: "Kansai",            hold: 18 },
    { id: "chugoku",   name: "Chugoku",           hold: 18 },
    { id: "shikoku",   name: "Shikoku",           hold: 18 },
    { id: "kyushu",    name: "Kyushu",            hold: 18 },
    { id: "okinawa",   name: "Okinawa",           hold: 18 },
    { id: null,        name: "All Japan (close)",  hold: 30 },
];

function ensureDir(dir) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function cleanDir(dir) {
    if (fs.existsSync(dir)) fs.rmSync(dir, { recursive: true });
    fs.mkdirSync(dir, { recursive: true });
}

async function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}

async function main() {
    ensureDir(OUT_DIR);
    cleanDir(FRAME_DIR);

    console.log("Launching browser...");
    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext({
        viewport: { width: WIDTH, height: HEIGHT },
        deviceScaleFactor: 2,
    });
    const page = await context.newPage();

    console.log(`Loading ${SITE_URL}`);
    await page.goto(SITE_URL, { waitUntil: "networkidle" });
    await sleep(3000);

    // Switch to area tab
    console.log("Switching to area tab...");
    await page.click('[data-tab="tab-area"]');
    await sleep(2000);

    // Hide substations and plants
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

    // Hide sidebar
    await page.evaluate(() => {
        var sidebar = document.getElementById("sidebar");
        if (sidebar) sidebar.style.display = "none";
        if (typeof map !== "undefined" && map.invalidateSize) map.invalidateSize();
    });
    await sleep(1000);

    let frameNum = 0;
    const pad = (n) => String(n).padStart(5, "0");

    const capture = async () => {
        await page.screenshot({
            path: path.join(FRAME_DIR, `frame_${pad(frameNum++)}.png`),
        });
    };

    for (const region of REGIONS) {
        console.log(`  ${region.name}...`);

        if (region.id) {
            // Zoom into region
            await page.evaluate((rid) => {
                if (typeof selectRegion === "function") selectRegion(rid);
            }, region.id);
        } else {
            // All-Japan: zoom out to show Hokkaido–Okinawa fully
            await page.evaluate(() => {
                if (typeof selectRegion === "function") selectRegion(null);
                if (typeof map !== "undefined") map.setView([35.5, 136.0], 5);
            });
        }

        // Capture zoom animation frames
        for (let i = 0; i < 10; i++) {
            await sleep(FRAME_DELAY_MS);
            await capture();
        }

        // Wait for tiles to load
        await sleep(2000);

        // Hold frames on final view
        for (let i = 0; i < region.hold; i++) {
            await capture();
            await sleep(FRAME_DELAY_MS);
        }
    }

    await browser.close();

    // Build GIF
    console.log("Building GIF...");
    const outPath = path.join(OUT_DIR, "network_tour.gif");
    const palette = path.join(FRAME_DIR, "palette.png");
    execSync(
        `ffmpeg -y -framerate ${GIF_FPS} -i "${FRAME_DIR}/frame_%05d.png" ` +
        `-vf "palettegen=max_colors=128:stats_mode=diff" "${palette}"`,
        { stdio: "pipe" }
    );
    execSync(
        `ffmpeg -y -framerate ${GIF_FPS} -i "${FRAME_DIR}/frame_%05d.png" ` +
        `-i "${palette}" -lavfi "paletteuse=dither=bayer:bayer_scale=3" ` +
        `-loop 0 "${outPath}"`,
        { stdio: "pipe" }
    );

    // Cleanup
    fs.rmSync(FRAME_DIR, { recursive: true });

    const size = (fs.statSync(outPath).size / 1024 / 1024).toFixed(1);
    console.log(`\nDone! ${outPath} (${size} MB)`);
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
