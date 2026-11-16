// sctag: makes the SoundCloud playlists you picked. Paste it into the console of soundcloud.com, logged in.
// It only creates private playlists titled "sctag · ..." and replaces the track list of those playlists (yours, with
// that title). It never deletes anything and never touches likes, follows, reposts or your other playlists.
// Written by `sctag playlists --browser` (template: sctag/playlists_browser.js).
var sctagDone = (async () => {
  const PLAN = /*PLAN*/null; // {client_id, prefix, pauses: {read: [s, s], write: [s, s]}, playlists: {title: [ids]}}
  const API = "https://api-v2.soundcloud.com";
  const STORE = "sctag-playlists"; // title -> id of the playlists this script made, remembered in this browser
  const WRITES = [["POST", /^\/playlists$/], ["PUT", /^\/playlists\/\d+$/]]; // all this script will ever write
  const log = (...a) => console.log("sctag:", ...a);

  class Stop extends Error {}
  let last = 0;
  const pause = async ([lo, hi]) => {
    const wait = (lo + Math.random() * (hi - lo)) * 1000 - (Date.now() - last);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    last = Date.now();
  };

  const cookie = (document.cookie.match(/(?:^|; )oauth_token=([^;]+)/) || [])[1];
  const token = cookie ? decodeURIComponent(cookie) : prompt("sctag: paste your oauth_token (it isn't saved)");
  if (!token) throw new Stop("not logged in: log in on soundcloud.com, reload, and paste the script again.");

  async function request(method, path, body) {
    await pause(method === "GET" ? PLAN.pauses.read : PLAN.pauses.write);
    const url = path.startsWith("https://") ? path : API + path;
    const res = await fetch(`${url}${url.includes("?") ? "&" : "?"}client_id=${PLAN.client_id}`, {
      method,
      credentials: "include",
      headers: { Authorization: `OAuth ${token}`, Accept: "application/json", ...(body && { "Content-Type": "application/json" }) },
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    if (res.ok) return text ? JSON.parse(text) : null;
    if (res.status === 404 && method === "GET") return null;
    if (res.status === 429) throw new Stop("SoundCloud says too many requests: wait an hour, then paste it again.");
    if (res.status === 403 && /captcha-delivery|datadome/i.test(text))
      throw new Stop("SoundCloud's bot protection blocked it. If the page shows you a check, solving it yourself is fine; then paste it again.");
    throw new Stop(`${method} ${path.split("?")[0]} failed: HTTP ${res.status} ${text.slice(0, 200)}`);
  }

  async function write(method, path, body) {
    if (!WRITES.some(([m, rx]) => m === method && rx.test(path)))
      throw new Stop(`refusing to ${method} ${path}: sctag only creates and edits its own playlists`);
    return request(method, path, body);
  }

  const me = await request("GET", "/me");
  if (!me) throw new Stop("not logged in: log in on soundcloud.com, reload, and paste the script again.");
  const ours = (pl) => !!pl && (pl.user_id ?? pl.user?.id) === me.id && (pl.title || "").startsWith(PLAN.prefix);

  async function create(title, tracks) {
    if (!title.startsWith(PLAN.prefix)) throw new Stop(`refusing to create ${title}: not an sctag title`);
    const playlist = { title, sharing: "private", tracks, _resource_type: "playlist" };
    return write("POST", "/playlists", { playlist });
  }

  async function setTracks(pl, tracks) {
    if (!ours(pl)) throw new Stop(`refusing to edit ${pl.title}: not an sctag playlist of yours`); // checked again
    return write("PUT", `/playlists/${pl.id}`, { playlist: { tracks } });
  }

  // playlists made earlier: remembered in this browser, else found on your profile by title
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORE)) || {};
  } catch {}
  const known = {};
  try {
    let next = `/users/${me.id}/playlists_without_albums?limit=200&linked_partitioning=1`;
    while (next) {
      const page = await request("GET", next);
      for (const pl of page?.collection || []) if (ours(pl)) known[pl.title] ??= pl.id;
      next = page?.next_href;
    }
  } catch (e) {
    if (e instanceof Stop && /bot protection|too many/.test(e.message)) throw e;
    log("could not list your playlists, using the ones remembered in this browser:", e.message);
  }
  Object.assign(known, saved);

  const todo = [];
  for (const [title, want] of Object.entries(PLAN.playlists)) {
    const pl = known[title] ? await request("GET", `/playlists/${known[title]}?representation=full`) : null;
    if (pl && !ours(pl)) log(`skip ${title}: playlist ${pl.id} isn't yours or was renamed, not touching it`);
    else if (!pl) todo.push(["create", title, want, null]);
    else if (JSON.stringify((pl.tracks || []).map((t) => t.id)) !== JSON.stringify(want)) todo.push(["update", title, want, pl]);
    else log(`ok ${title}: up to date`);
  }
  const stale = Object.keys(known).filter((t) => !(t in PLAN.playlists));
  if (stale.length) log("not picked anymore, left untouched (delete them in the app if you like):", stale.join(", "));
  if (!todo.length) return log("nothing to change");

  const minutes = Math.max(1, Math.round((todo.length * (PLAN.pauses.write[0] + PLAN.pauses.write[1])) / 2 / 60));
  const list = todo.map(([action, title, want]) => `${action} ${title} (${want.length})`).join("\n");
  if (!confirm(`sctag, logged in as ${me.username}: make these changes? (about ${minutes} min, keep the tab open)\n\n${list}`))
    return log("nothing changed");

  for (const [action, title, want, pl] of todo) {
    log(`${action} ${title} ...`);
    const out = action === "create" ? await create(title, want) : await setTracks(pl, want);
    saved[title] = out.id;
    try {
      localStorage.setItem(STORE, JSON.stringify(saved));
    } catch {}
    const got = out.track_count ?? want.length;
    log(`${action}d ${title}` + (got === want.length ? "" : ` (SoundCloud kept ${got} of ${want.length}: some are unavailable)`));
    if (got < want.length / 2) throw new Stop(`SoundCloud kept only ${got} of ${want.length} tracks in ${title}`);
  }
  log("done: your playlists are in Library > Playlists, private.");
})().catch((e) => console.error("sctag stopped:", e.message, "Nothing more was changed."));
