from sctag.text_tags import artist_credits, norm, rappers_in, text_tags


def tags(title="", genre="", tag_list="", artist="someone", kind="tracks"):
    return dict(text_tags({"title": title, "sc_genre": genre, "sc_tags": tag_list, "artist": artist}, kind))


def test_norm_strips_accents_and_dollars():
    assert norm("  Funk MANDELÃO  A$AP ") == "funk mandelao asap"


def test_genre_from_title_genre_field_and_tags():
    assert tags(title="Rock Doido Mix") == {"rock doido": "title", "baile funk": "title"}
    assert tags(genre="Ambient") == {"ambient": "genre"}
    assert tags(tag_list="jersey club bass") == {"jersey club": "tags"}


def test_first_source_wins():
    assert tags(title="techno edit", tag_list="techno")["techno"] == "title"


def test_brazilian_subgenres_imply_baile_funk():
    assert tags(title="bruxaria 2025")["baile funk"] == "title"


def test_tag_stuffing_is_ignored():
    stuffed = "Dembow Reggaeton Hardstyle Techno Experimental"
    assert tags(tag_list=stuffed) == {}
    assert tags(title="techno", tag_list=stuffed) == {"techno": "title"}


def test_rapper_in_title_is_a_rap_edit_unless_they_uploaded_it():
    assert tags(title="Chief Keef - Citgo (club blend)")["rap edit"] == "title"
    assert tags(title="Chief Keef - Citgo", artist="Chief Keef") == {"rap": "title"}
    assert "rap edit" not in tags(title="Chief Keef - Citgo (mix)", kind="sets")


def test_ambiguous_names_only_count_as_whole_credits():
    assert rappers_in("Future - Mask Off (edit)") == {"future"}
    assert rappers_in("the future of club music") == set()
    assert artist_credits("Drake x Future feat. Young Thug - Song") >= {"drake", "future", "young thug"}
