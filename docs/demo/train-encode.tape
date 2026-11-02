# Records docs/demo/train-encode.gif from a real run: `vhs docs/demo/train-encode.tape` from the repo root.
Output docs/demo/train-encode.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 460
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m bpe_tokenizer_library train` Enter
Sleep 2.5s
Type `python -m bpe_tokenizer_library encode "the dog"` Enter
Sleep 2.5s
