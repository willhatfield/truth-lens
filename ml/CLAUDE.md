Always use context7 when I need code generation, setup or configuration steps, or library/API documentation. This means you should automatically use the Context7 MCP tools to resolve library id and get library docs without me having to explicitly ask.

Always do all new work on a new branch, without me having to explicitly ask.

Restrict all code to very simple control flow constructs—do not use goto statements, setjmp or longjmp constructs, or direct or indirect recursion. Give all loops a fixed upper bound. No function should be longer than what can be printed on a single sheet of paper in a standard format with one line per statement and one line per declaration. Declare all data objects at the smallest possible level of scope. Limit pointer use to a single dereference, and do not use function pointers. All new functions / features / bug fixes need tests that target the specified functionality.

Any componenents that might be reused accross different files should be created in their own file.

In a seperate subfolder called ClaudeProgress, save a .MD file for changes made, bugs encountered, and next steps for the
 next agent to follow up on
