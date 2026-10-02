// Psychophysics for Machines — design mockup behaviour
// 1. the Article / Notebook flip, landing at the same section
// 2. the list of parts highlights the section being read

(function () {
  var params = new URLSearchParams(window.location.search);

  // ---------- the crowded title ----------
  // Wrap every letter in its own span and give it a small fixed offset and tilt,
  // so the title looks jumbled until it is hovered (or tapped on a phone).
  var title = document.querySelector(".crowded-title");
  if (title) {
    title.setAttribute("aria-label", title.textContent);

    // a tiny random-number generator with a fixed seed, so the jumble is the same on every visit
    var seed = 7;
    function random() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    var parts = title.innerHTML.split("<br>");
    var html  = "";
    for (var p = 0; p < parts.length; p++) {
      var text = parts[p];
      for (var c = 0; c < text.length; c++) {
        var ch = text[c];
        if (ch === " ") {
          html = html + " ";
          continue;
        }
        var dx = ((random() - 0.5) * 0.3).toFixed(3) + "em";
        var dy = ((random() - 0.5) * 0.3).toFixed(3) + "em";
        var r  = ((random() - 0.5) * 34).toFixed(1) + "deg";
        html = html + '<span class="ch" aria-hidden="true" style="--dx:' + dx + ';--dy:' + dy + ';--r:' + r + '">' + ch + "</span>";
      }
      if (p < parts.length - 1) html = html + "<br>";
    }
    title.innerHTML = html;

    title.addEventListener("click", function () {
      title.classList.toggle("released");
    });
  }

  // ---------- the flip ----------
  var viewSwitch = document.querySelector(".view-switch");
  var faces      = { article: document.getElementById("article"), notebook: document.getElementById("notebook") };
  var tocLinks   = document.querySelectorAll(".toc a");
  var busy       = false;
  var savedScroll = {};   // scroll position of each view when we left it

  function currentView() {
    return viewSwitch ? viewSwitch.getAttribute("data-view") : "article";
  }

  // the section whose top is closest above the middle of the screen
  function sectionInView(face) {
    var sections = face.querySelectorAll("section[data-section]");
    var best     = null;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].getBoundingClientRect().top < window.innerHeight * 0.4) {
        best = sections[i];
      }
    }
    return best ? best.getAttribute("data-section") : null;
  }

  function showView(name, animate) {
    if (!viewSwitch || busy || name === currentView()) return;
    var from    = faces[currentView()];
    var to      = faces[name];
    var section = sectionInView(from);
    savedScroll[currentView()] = window.scrollY;   // remember where we left this view

    function swap() {
      from.hidden = true;
      from.classList.remove("flip-out");
      to.hidden = false;
      viewSwitch.setAttribute("data-view", name);
      var buttons = viewSwitch.querySelectorAll("button");
      for (var i = 0; i < buttons.length; i++) {
        buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-view") === name ? "true" : "false");
      }

      // coming back to a view we were already reading: return to the same place
      if (savedScroll[name] !== undefined) {
        window.scrollTo({ top: savedScroll[name], behavior: "instant" });
      } else {
        // first visit: land on the same section; the notebook has no parts 1-2, so fall back to part 3
        var target = section ? to.querySelector('section[data-section="' + section + '"]') : null;
        if (!target && section) target = to.querySelector("section[data-section]");
        if (target) target.scrollIntoView({ behavior: "instant", block: "start" });
      }

      if (animate) {
        to.classList.add("flip-in");
        to.addEventListener("animationend", function done() {
          to.classList.remove("flip-in");
          to.removeEventListener("animationend", done);
          busy = false;
        });
      } else {
        busy = false;
      }
      updateToc();
    }

    busy = true;
    if (animate) {
      from.classList.add("flip-out");
      from.addEventListener("animationend", function done() {
        from.removeEventListener("animationend", done);
        swap();
      });
    } else {
      swap();
    }
  }

  if (viewSwitch) {
    viewSwitch.addEventListener("click", function (event) {
      var button = event.target.closest("button");
      if (button) showView(button.getAttribute("data-view"), true);
    });
    if (params.get("view") === "notebook") showView("notebook", false);
  }

  // ---------- table of contents ----------
  function updateToc() {
    if (!tocLinks.length) return;
    var face    = faces[currentView()] || document;
    var current = sectionInView(face);
    for (var i = 0; i < tocLinks.length; i++) {
      var link = tocLinks[i];
      var id   = link.getAttribute("data-section");
      link.classList.toggle("active", id === current);
      link.classList.toggle("no-code", currentView() === "notebook" && !faces.notebook.querySelector('section[data-section="' + id + '"]'));
    }
  }

  for (var i = 0; i < tocLinks.length; i++) {
    tocLinks[i].addEventListener("click", function (event) {
      event.preventDefault();
      var face   = faces[currentView()];
      var target = face.querySelector('section[data-section="' + this.getAttribute("data-section") + '"]');
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  window.addEventListener("scroll", updateToc, { passive: true });
  updateToc();
})();
