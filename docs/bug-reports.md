# Plottle — Bug Reports

> **Inbox for user-reported bugs.**
>
> Add new bugs under `## Unresolved` at any time — mid-session, between sessions,
> whenever you notice something. No special format required; a clear description
> of what happened is enough.
>
> The AI agent processes this file at the end of every session as part of the
> Closing Protocol: items in `## Unresolved` get routed to the task system and
> moved verbatim to `## Incorporated into TODO`. Confirmed fixes move to `## Resolved`.

---

## Unresolved

────────────────────────── Traceback (most recent call last) ───────────────────────────
  C:\Users\schul\Documents\local-code-dir\repos-PUBLIC\Plottle\.venv\Lib\site-packages
  \streamlit\runtime\scriptrunner\exec_code.py:129 in exec_func_with_error_handling

  C:\Users\schul\Documents\local-code-dir\repos-PUBLIC\Plottle\.venv\Lib\site-packages
  \streamlit\runtime\scriptrunner\script_runner.py:689 in code_to_exec

  C:\Users\schul\Documents\local-code-dir\repos-PUBLIC\plottle\modules\Home.py:393 in
  <module>

    390
    391 # ── Execute the active page ─────────────────────────────────────────────────
    392
  ❱ 393 pg.run()
    394

  C:\Users\schul\Documents\local-code-dir\repos-PUBLIC\Plottle\.venv\Lib\site-packages
  \streamlit\navigation\page.py:380 in run

  C:\Users\schul\Documents\local-code-dir\repos-PUBLIC\plottle\modules\pages\2_Quick_P
  lot.py:1468 in <module>

    1465 │   │   # Handle newly selected points
    1466 │   │   if _event and hasattr(_event, "selection"):
    1467 │   │   │   for _pt in _event.selection.points:
  ❱ 1468 │   │   │   │   _ann = {"x": _pt.x, "y": _pt.y}
    1469 │   │   │   │   if _ann not in st.session_state.qp_annotations:
    1470 │   │   │   │   │   st.session_state.qp_annotations.append(_ann)
    1471 │   except TypeError:
────────────────────────────────────────────────────────────────────────────────────────
AttributeError: 'dict' object has no attribute 'x'


<!-- Add new bugs here. One bullet per bug. Helpful to include:
     - What you did (the action or scenario that triggered it)
     - What happened (the actual behavior)
     - What you expected (the intended behavior)
     Example:
       * Clicking "Save" while a text field is empty throws an unhandled exception
         instead of showing a validation message -->

*(no unresolved bugs)*

---

## Incorporated into TODO

<!-- Items moved here verbatim from Unresolved after being added to
     GOTCHAS.md or a roadmap/backlog. Do not edit original text.
     Items remain here until the fix is confirmed. -->

*(none yet)*

---

## Resolved

<!-- Items moved here from "Incorporated into TODO" when the fix is confirmed.
     Format: original bullet + fix note
     Example:
       * Clicking "Save" while a text field is empty throws an unhandled exception
         — fixed YYYY-MM-DD: added input validation in form_handler.py -->

*(none yet)*
