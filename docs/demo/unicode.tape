# Records docs/demo/unicode.gif from a real run: `vhs docs/demo/unicode.tape` from the repo root.
Output docs/demo/unicode.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 440
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate` Enter
Sleep 2s
Sleep 10s
Type `clear` Enter
Sleep 1s
Show
Type `python -m bpe_tokenizer_library encode "naïve café"` Enter
Sleep 2.5s
Type `python -m bpe_tokenizer_library encode "東京 😀"` Enter
Sleep 3s
