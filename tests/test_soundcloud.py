from sctag.soundcloud import merge, skip_before, slim_track


def track(i, **kw):
    return {"id": i, "title": f"t{i}", **kw}


def test_merge_marks_sources_and_keeps_newest_first():
    likes = [track(1, liked_at="2024-05-01"), track(2, liked_at="2023-01-01")]
    uploads = [track(2, uploaded_at="2019-01-01"), track(3, uploaded_at="2023-06-01")]
    reposts = [track(3, reposted_at="2024-01-01"), track(4, reposted_at="2022-01-01")]
    out = merge(likes, uploads, reposts)
    assert [t["id"] for t in out] == [1, 3, 2, 4]  # unliked ones sort by upload / repost date
    by_id = {t["id"]: t for t in out}
    assert by_id[1] == likes[0]
    assert by_id[2]["posted"] and "liked" not in by_id[2]  # liked and posted: still a like
    assert by_id[3] == {
        **uploads[1],
        "posted": True,
        "reposted": True,
        "reposted_at": "2024-01-01",
        "liked": False,
        "liked_at": "2023-06-01",
    }
    assert by_id[4] == {**reposts[1], "reposted": True, "liked": False, "liked_at": "2022-01-01"}


def test_skip_before_cuts_old_likes_and_reposts_but_keeps_posts():
    tracks = [
        track(1, liked_at="2024-01-01T10:00:00Z"),
        track(2, liked_at="2015-01-01T10:00:00Z"),
        track(3, liked_at="2015-01-01T10:00:00Z", reposted=True, reposted_at="2014-01-01T10:00:00Z"),
        track(4, liked_at="2015-01-01T10:00:00Z", reposted=True, reposted_at="2020-01-01T10:00:00Z"),
        track(5, liked_at="2013-01-01T10:00:00Z", reposted_at="2013-01-01T10:00:00Z", reposted=True, liked=False),
        track(6, liked_at="2014-01-01T10:00:00Z", posted=True, liked=False),
    ]
    assert [t["id"] for t in skip_before(tracks, "2016-10-27")] == [1, 4, 6]
    assert skip_before(tracks, None) == tracks


def test_slim_track_keeps_both_dates():
    t = slim_track({"id": 1, "created_at": "2024-01-02T00:00:00Z", "release_date": "2019-05-01T00:00:00Z", "user": {}})
    assert (t["uploaded_at"], t["release_date"]) == ("2024-01-02T00:00:00Z", "2019-05-01T00:00:00Z")
