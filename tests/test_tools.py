"""One happy path and one ambiguity case per tool (CLAUDE.md, "Adding a tool").

Anchored to Thursday 6 August 2026, so COMP4020 -- a Thursday class -- is on
today, and "next friday" is the fourteenth.
"""

from __future__ import annotations

import pytest


def clarifies(result) -> bool:
    return result.get("needs") == "clarification"


def confirms(result) -> bool:
    return result.get("needs") == "confirmation"


# -- get_now ---------------------------------------------------------------


def test_get_now_uses_the_injected_clock(call):
    result = call("get_now")
    assert result["date"] == "2026-08-06"
    assert result["weekday"] == "Thursday"
    assert result["time"] == "10:00"


# -- list_schedule ---------------------------------------------------------


def test_list_schedule_today(call, a_course):
    result = call("list_schedule", when="today")
    assert result["count"] == 1
    assert result["items"][0]["code"] == "COMP4020"
    assert result["items"][0]["start"] == "09:00"


def test_list_schedule_expands_a_week(call, a_course):
    result = call("list_schedule", when="next week")
    assert result["count"] == 1
    assert result["items"][0]["date"] == "2026-08-13"


def test_list_schedule_respects_term_dates(call):
    call(
        "add_class", code="COMP3500", weekday="thursday", start="2pm", end="4pm",
        term="1 september to 30 november",
    )
    assert call("list_schedule", when="today")["count"] == 0
    assert call("list_schedule", when="3 september")["count"] == 1


def test_list_schedule_applies_a_cancellation(call, a_course):
    call("cancel_class", course="COMP4020", date="today", kind="cancelled")
    result = call("list_schedule", when="today")
    assert result["items"][0]["status"] == "cancelled"


def test_list_schedule_applies_a_room_change(call, a_course):
    call(
        "cancel_class", course="COMP4020", date="today",
        kind="room_change", new_location="Chifley Library",
    )
    assert call("list_schedule", when="today")["items"][0]["location"] == "Chifley Library"


def test_list_schedule_will_not_invent_term_dates(call):
    assert clarifies(call("list_schedule", when="this semester"))


# -- list_assignments ------------------------------------------------------


def test_list_assignments_hides_finished_work_by_default(call):
    call("add_assignment", title="essay", due="next friday")
    call("add_assignment", title="lab report", due="tomorrow")
    call("set_progress", assignment="lab report", percent=100)

    result = call("list_assignments")
    assert [item["title"] for item in result["items"]] == ["essay"]
    assert call("list_assignments", status="all")["count"] == 2
    assert call("list_assignments", status="done")["count"] == 1


def test_list_assignments_sorts_by_due_date(call):
    call("add_assignment", title="later", due="next friday")
    call("add_assignment", title="sooner", due="tomorrow")
    assert [i["title"] for i in call("list_assignments")["items"]] == ["sooner", "later"]


def test_list_assignments_filters_by_due_date_and_course(call, a_course):
    call("add_assignment", title="essay", due="next friday", course="COMP4020")
    call("add_assignment", title="far off", due="in two months")
    assert call("list_assignments", due_before="next friday")["count"] == 1
    assert call("list_assignments", course="COMP4020")["count"] == 1


def test_due_before_accepts_a_span_not_just_a_day(call):
    """"due this week" is how people say it; it must not raise "which day?"."""
    call("add_assignment", title="soon", due="tomorrow")
    call("add_assignment", title="later", due="in three weeks")
    result = call("list_assignments", due_before="this week")
    assert [i["title"] for i in result["items"]] == ["soon"]
    assert result["filter"]["due_before"] == "this week"


def test_list_assignments_derives_hours_left(call):
    call("add_assignment", title="essay", due="next friday", est_hours=6)
    call("set_progress", assignment="essay", percent=25)
    assert call("list_assignments")["items"][0]["hours_left"] == 4.5


def test_list_assignments_rejects_an_unknown_status(call):
    assert clarifies(call("list_assignments", status="pending"))


# -- add_class -------------------------------------------------------------


def test_add_class_happy_path(call, a_course, conn):
    assert a_course["weekday"] == "Thursday"
    assert a_course["start"] == "09:00"
    row = conn.execute("SELECT * FROM course WHERE id = ?", (a_course["id"],)).fetchone()
    assert row["code"] == "COMP4020"
    assert row["deleted_at"] is None


def test_add_class_with_a_term(call):
    call(
        "add_class", code="ENGN4122", weekday="monday", start="10am", end="12pm",
        term="18 august to 27 october",
    )
    row = call("list_schedule", when="24 august")["items"][0]
    assert row["code"] == "ENGN4122"


def test_add_class_backwards_times_ask(call):
    result = call("add_class", code="COMP4620", weekday="monday", start="4pm", end="2pm")
    assert clarifies(result)
    assert "backwards" in result["question"]


def test_add_class_bare_hour_asks(call):
    assert clarifies(call("add_class", code="COMP4620", weekday="monday", start="9", end="11"))


def test_add_class_duplicate_code_asks_rather_than_silently_doubling(call, a_course):
    result = call("add_class", code="COMP4020", weekday="tuesday", start="1pm", end="3pm")
    assert clarifies(result)
    assert "second session" in result["question"]


def test_add_class_missing_argument_asks(call):
    assert clarifies(call("add_class", code="COMP4620", weekday="monday", start="9am"))


# -- update_class ----------------------------------------------------------


def test_update_class_non_destructive_field_goes_straight_through(call, a_course, conn):
    result = call("update_class", course="COMP4020", location="Copland G30")
    assert result["updated"] == "course"
    row = conn.execute("SELECT * FROM course WHERE id = ?", (a_course["id"],)).fetchone()
    assert row["location"] == "Copland G30"


def test_update_class_overwriting_a_time_confirms_first(call, a_course, conn):
    result = call("update_class", course="COMP4020", start="10am")
    assert confirms(result)
    row = conn.execute("SELECT start_time FROM course WHERE id = ?", (a_course["id"],)).fetchone()
    assert row["start_time"] == "09:00", "nothing may change before the user says yes"

    resumed = result["resume"]
    after = call(resumed["tool"], confirmed=True, **resumed["args"])
    assert after["start"] == "10:00"


def test_update_class_unknown_field_asks(call, a_course):
    assert clarifies(call("update_class", course="COMP4020", vibe="chill"))


def test_update_class_ambiguous_course_asks(call, a_course):
    call("add_class", code="COMP4620", weekday="monday", start="1pm", end="3pm")
    result = call("update_class", course="comp", location="somewhere")
    assert clarifies(result)
    assert len(result.get("options", [])) >= 2


# -- cancel_class ----------------------------------------------------------


def test_cancel_class_happy_path(call, a_course):
    result = call("cancel_class", course="COMP4020", date="today", kind="cancelled")
    assert result["kind"] == "cancelled"
    assert result["date"] == "2026-08-06"


def test_cancel_class_on_a_day_it_does_not_meet_asks(call, a_course):
    result = call("cancel_class", course="COMP4020", date="tomorrow", kind="cancelled")
    assert clarifies(result)
    assert "Friday" in result["question"]


def test_moved_class_keeps_its_duration(call, a_course):
    result = call(
        "cancel_class", course="COMP4020", date="today", kind="moved", new_start="1pm"
    )
    assert result["start"] == "13:00"
    assert result["end"] == "15:00"


def test_moved_class_without_a_new_time_asks(call, a_course):
    assert clarifies(call("cancel_class", course="COMP4020", date="today", kind="moved"))


def test_room_change_without_a_room_asks(call, a_course):
    assert clarifies(call("cancel_class", course="COMP4020", date="today", kind="room_change"))


# -- delete_class ----------------------------------------------------------


def test_delete_class_confirms_then_deletes(call, a_course, conn):
    first = call("delete_class", course="COMP4020")
    assert confirms(first)
    assert conn.execute("SELECT deleted_at FROM course").fetchone()[0] is None

    call("delete_class", confirmed=True, **first["resume"]["args"])
    assert conn.execute("SELECT deleted_at FROM course").fetchone()[0] is not None


def test_delete_class_unknown_course_asks(call, a_course):
    assert clarifies(call("delete_class", course="basket weaving"))


# -- add_assignment --------------------------------------------------------


def test_add_assignment_happy_path(call):
    result = call("add_assignment", title="data structures", due="next friday", est_hours=6)
    assert result["due_date"] == "2026-08-14"
    assert result["explicit_time"] is False
    assert result["est_hours"] == 6.0


def test_add_assignment_takes_spoken_hours(call):
    result = call("add_assignment", title="essay", due="tomorrow", est_hours="about six hours")
    assert result["est_hours"] == 6.0


def test_add_assignment_links_a_course(call, a_course):
    result = call("add_assignment", title="essay", due="tomorrow", course="COMP4020")
    assert result["course"].startswith("COMP4020")


def test_an_unknown_course_does_not_sink_the_write(call):
    """The course is optional, so a name that matches nothing is not a refusal."""
    result = call("add_assignment", title="data structures", due="next friday", course="data structures")
    assert result["created"] == "assignment"
    assert result["course"] is None
    assert result["unlinked_course"] == "data structures"


def test_an_ambiguous_course_still_asks_even_though_it_is_optional(call, a_course):
    call("add_class", code="COMP4620", weekday="monday", start="1pm", end="3pm")
    assert clarifies(call("add_assignment", title="essay", due="tomorrow", course="comp"))


def test_add_assignment_vague_due_date_asks(call):
    result = call("add_assignment", title="essay", due="sometime")
    assert clarifies(result)
    assert result["question"].endswith("?")


def test_add_assignment_without_a_due_date_asks(call):
    assert call("add_assignment", title="essay")["question"] == "When?"


def test_add_assignment_flags_a_past_due_date(call):
    assert call("add_assignment", title="late thing", due="yesterday")["overdue"] is True


# -- update_assignment -----------------------------------------------------


def test_update_assignment_non_destructive_change(call):
    call("add_assignment", title="essay", due="next friday")
    result = call("update_assignment", assignment="essay", est_hours=4)
    assert "est_hours" in result["changed"]


def test_update_assignment_moving_a_due_date_confirms_first(call, conn):
    call("add_assignment", title="essay", due="next friday")
    result = call("update_assignment", assignment="essay", due="tomorrow")
    assert confirms(result)
    assert conn.execute("SELECT due_at FROM assignment").fetchone()[0].startswith("2026-08-14")

    resumed = result["resume"]
    after = call(resumed["tool"], confirmed=True, **resumed["args"])
    assert after["due_date"] == "2026-08-07"


def test_update_assignment_unknown_assignment_asks(call):
    call("add_assignment", title="essay", due="tomorrow")
    assert clarifies(call("update_assignment", assignment="quantum knitting", est_hours=2))


# -- set_progress ----------------------------------------------------------


def test_set_progress_moves_status_along(call):
    call("add_assignment", title="essay", due="next friday", est_hours=6)
    result = call("set_progress", assignment="essay", percent=60)
    assert result["progress_pct"] == 60
    assert result["status"] == "in_progress"
    assert result["hours_left"] == 2.4


def test_set_progress_to_a_hundred_marks_it_done(call):
    call("add_assignment", title="essay", due="next friday")
    assert call("set_progress", assignment="essay", percent=100)["status"] == "done"


def test_set_progress_takes_spoken_percentages(call):
    call("add_assignment", title="essay", due="next friday")
    assert call("set_progress", assignment="essay", percent="sixty percent")["progress_pct"] == 60


@pytest.mark.parametrize(
    "spoken,expected", [("50%", 50), ("fifty", 50), ("ninety percent", 90), ("a hundred", 100)]
)
def test_set_progress_understands_how_people_say_numbers(call, spoken, expected):
    call("add_assignment", title="essay", due="next friday")
    assert call("set_progress", assignment="essay", percent=spoken)["progress_pct"] == expected


def test_set_progress_out_of_range_asks(call):
    call("add_assignment", title="essay", due="next friday")
    assert clarifies(call("set_progress", assignment="essay", percent=180))


# -- delete_assignment -----------------------------------------------------


def test_delete_assignment_confirms_then_deletes(call, conn):
    call("add_assignment", title="essay", due="next friday")
    first = call("delete_assignment", assignment="essay")
    assert confirms(first)
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None

    call("delete_assignment", confirmed=True, **first["resume"]["args"])
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is not None


def test_delete_assignment_ambiguous_target_asks(call):
    call("add_assignment", title="essay one", due="tomorrow")
    call("add_assignment", title="essay two", due="tomorrow")
    assert clarifies(call("delete_assignment", assignment="essay"))


# -- add_reminder ----------------------------------------------------------


def test_add_reminder_defaults_to_the_morning(call):
    result = call("add_reminder", title="print the notes", when="tomorrow")
    assert result["date"] == "2026-08-07"
    assert result["when"] == "2026-08-07T09:00:00+10:00"
    assert result["explicit_time"] is False


def test_add_reminder_links_to_an_assignment(call, conn):
    call("add_assignment", title="essay", due="next friday")
    call("add_reminder", title="start the essay", when="tomorrow", related="essay")
    row = conn.execute("SELECT * FROM reminder").fetchone()
    assert row["related_type"] == "assignment"


def test_add_reminder_vague_time_asks(call):
    assert clarifies(call("add_reminder", title="something", when="soon"))


# -- undo_last_write -------------------------------------------------------


def test_undo_reverses_the_last_write(call, conn):
    call("add_assignment", title="essay", due="next friday")
    assert call("undo_last_write")["undone"] == 1
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is not None


def test_undo_with_nothing_to_undo_says_so(call):
    result = call("undo_last_write")
    assert "nothing to undo" in result["error"]


def test_undo_after_a_confirmed_delete(call, conn):
    call("add_assignment", title="essay", due="next friday")
    pending = call("delete_assignment", assignment="essay")
    call("delete_assignment", confirmed=True, **pending["resume"]["args"])

    call("undo_last_write")
    assert conn.execute("SELECT deleted_at FROM assignment").fetchone()[0] is None


# -- fuzzy matching --------------------------------------------------------


def test_exact_code_wins_over_a_near_neighbour(call, a_course):
    call("add_class", code="COMP4620", weekday="monday", start="1pm", end="3pm")
    assert call("update_class", course="COMP4020", location="X")["code"] == "COMP4020"


def test_spoken_digits_match_a_course_code(call, a_course):
    result = call("update_class", course="comp four zero two zero", location="X")
    assert result.get("code") == "COMP4020"


def test_matching_by_title_fragment(call, a_course):
    assert call("update_class", course="agentic coding", location="X")["code"] == "COMP4020"


def test_a_near_miss_course_code_is_never_a_silent_match(call, a_course):
    """COMP4620 is not COMP4020, however close the two sound."""
    result = call("cancel_class", course="COMP4620", date="today", kind="cancelled")
    assert clarifies(result)
    assert "could not find" in result["question"]


def test_two_plausible_matches_ask_rather_than_pick(call, a_course):
    call("add_class", code="COMP4620", weekday="monday", start="1pm", end="3pm")
    result = call("delete_class", course="comp4x20")
    assert clarifies(result)


def test_no_match_at_all_asks(call, a_course):
    result = call("delete_class", course="underwater basket weaving")
    assert clarifies(result)
    assert "could not find" in result["question"]


# -- registry ---------------------------------------------------------------


def test_every_planned_tool_is_registered():
    from src.tools import registry

    assert set(registry.TOOLS) == {
        "get_now", "list_schedule", "list_assignments", "add_class", "update_class",
        "cancel_class", "delete_class", "add_assignment", "update_assignment",
        "set_progress", "delete_assignment", "add_reminder", "undo_last_write",
    }


def test_only_writes_are_marked_destructive():
    from src.tools import registry

    for spec in registry.TOOLS.values():
        if spec.destructive:
            assert spec.writes, f"{spec.name} confirms but does not write"
    assert not registry.TOOLS["list_schedule"].writes
    assert not registry.TOOLS["get_now"].writes


def test_unknown_tool_asks_rather_than_raising(call):
    assert clarifies(call("make_coffee"))


@pytest.mark.parametrize("name", ["delete_class", "delete_assignment"])
def test_destructive_tools_are_flagged(name):
    from src.tools import registry

    assert registry.TOOLS[name].destructive
