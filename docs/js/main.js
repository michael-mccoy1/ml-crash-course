"use strict";
/* Entry point: load HLM data, build the first lesson, then load the rest in the background. */
(async function(){
  boot();
  try {
    progress(0.1);
    await loadDataset('HLM'); D = DATA.HLM; setRange(D); renderChips(); progress(0.5);
    await loadStructs(); progress(0.7);
    showLesson('start'); progress(1);
    loadExtras().catch(function(){});
  } catch (e) {
    $('lesson-start').innerHTML = '<div class="loading">Could not load the data: ' + esc(e.message) + '. Serve this folder over http (for example python -m http.server in docs).</div>';
    console.error(e);
  }
})();
