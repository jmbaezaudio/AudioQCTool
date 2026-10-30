# Presets

Place the Reaper session holding your dialed-in plugin chain here and point
`RPP_PATH` at it.

The pipeline reads plugin state directly from the `.RPP` file — dial settings
in Reaper, save, run. No preset export step.

The session must have all four plugins on the **Master FX chain**, in order:

1. FabFilter Pro-Q 3
2. iZotope RX 8 Voice De-noise  (instance 1)
3. iZotope RX 8 Voice De-noise  (instance 2)
4. Accentize dxRevive Pro

Audio items in the session are ignored, so there is no need to clean it up.

**Note:** `.rpp` files are gitignored by default, since plugin state encodes
your own tuning work. Remove that rule in `.gitignore` if you intend to publish
your chain.
