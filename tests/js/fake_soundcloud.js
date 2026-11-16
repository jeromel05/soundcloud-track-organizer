// Runs the browser script against a fake SoundCloud API. Usage: node fake_soundcloud.js script.js scenario.json
// Prints {calls, logs, store, playlists} as JSON.
const fs = require("fs");
const vm = require("vm");

const script = fs.readFileSync(process.argv[2], "utf8");
const scenario = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const playlists = scenario.playlists || {}; // id -> {id, user_id, title, tracks: [{id}]}
const store = scenario.store ? { "sctag-playlists": JSON.stringify(scenario.store) } : {};
const calls = [];
const logs = [];
let nextId = 100;

const reply = (status, body) => ({
  status,
  ok: status < 300,
  text: async () => (body === undefined ? "" : typeof body === "string" ? body : JSON.stringify(body)),
});

async function fetch(url, opts) {
  const u = new URL(url);
  const method = opts.method;
  const body = opts.body ? JSON.parse(opts.body) : null;
  calls.push({ method, path: u.pathname, body, auth: opts.headers.Authorization, client_id: u.searchParams.get("client_id") });
  if (scenario.block && method !== "GET") return reply(403, '{"url":"https://geo.captcha-delivery.com/captcha/?x=1"}');
  if (u.pathname === "/me") return reply(200, { id: 7, username: "me" });
  if (u.pathname === "/users/7/playlists_without_albums")
    return scenario.list_fails ? reply(500, "oops") : reply(200, { collection: Object.values(playlists), next_href: null });
  let m;
  if (method === "GET" && (m = u.pathname.match(/^\/playlists\/(\d+)$/)))
    return playlists[m[1]] ? reply(200, playlists[m[1]]) : reply(404);
  if (method === "POST" && u.pathname === "/playlists") {
    const kept = body.playlist.tracks.slice(0, scenario.keep ?? undefined);
    const pl = { id: nextId++, user_id: 7, title: body.playlist.title, sharing: body.playlist.sharing,
      tracks: kept.map((id) => ({ id })), track_count: kept.length };
    playlists[pl.id] = pl;
    return reply(201, pl);
  }
  if (method === "PUT" && (m = u.pathname.match(/^\/playlists\/(\d+)$/))) {
    const pl = playlists[m[1]];
    pl.tracks = body.playlist.tracks.map((id) => ({ id }));
    pl.track_count = pl.tracks.length;
    return reply(200, pl);
  }
  return reply(500, "unexpected");
}

const ctx = {
  fetch,
  URL,
  setTimeout,
  document: { cookie: "sc_anonymous_id=1; oauth_token=2-123-456-abc" },
  prompt: () => null,
  confirm: () => scenario.confirm ?? true,
  localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; } },
  console: { log: (...a) => logs.push(a.join(" ")), error: (...a) => logs.push(a.join(" ")) },
};
vm.createContext(ctx);
vm.runInContext(script, ctx);
ctx.sctagDone.then(() => {
  const saved = store["sctag-playlists"];
  console.log(JSON.stringify({ calls, logs, store: saved ? JSON.parse(saved) : {}, playlists }));
});
