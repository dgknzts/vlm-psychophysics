/* Playable figures. A Markdown line "[demo: bouma]" becomes <div data-demo="bouma">. */

(function () {
  var svgNS = "http://www.w3.org/2000/svg";
  var sloanLetters = "CDHKNORSVZ";

  function make(tag, attributes) {
    var element = document.createElementNS(svgNS, tag);
    for (var name in attributes) element.setAttribute(name, attributes[name]);
    return element;
  }

  function randomLetter(notThese) {
    var letter = notThese[0];
    while (notThese.indexOf(letter) !== -1) {
      letter = sloanLetters[Math.floor(Math.random() * sloanLetters.length)];
    }
    return letter;
  }

  /* Bouma demo: the cross and the middle letter stay put; a slider below the
     image moves the two flankers. The readout gives the spacing as a fraction
     of the eccentricity (distance from the cross to the middle letter). */
  function boumaDemo(box) {
    var width = 900, height = 220;
    var crossX  = 60;
    var targetX = 560, targetY = 110;
    var eccentricity = targetX - crossX;      // 500 px
    var smallest = 0.1, largest = 0.6;        // slider range, as a fraction of the eccentricity (0.1 = 50 px, letters just apart)

    var svg = make("svg", { viewBox: "0 0 " + width + " " + height, role: "img",
                            "aria-label": "Look at the cross. Use the slider to move the two letters beside the middle letter." });
    svg.appendChild(make("rect", { width: width, height: height, fill: "#ffffff" }));

    var cross = make("g", { stroke: "#000", "stroke-width": "3" });
    cross.appendChild(make("line", { x1: crossX - 15, x2: crossX + 15, y1: targetY, y2: targetY }));
    cross.appendChild(make("line", { x1: crossX, x2: crossX, y1: targetY - 15, y2: targetY + 15 }));
    svg.appendChild(cross);

    var letters = make("g", { "font-family": "Helvetica, Arial, sans-serif", "font-weight": "700", "font-size": "51",
                              fill: "#000", "text-anchor": "middle", "dominant-baseline": "central" });
    var left   = make("text", { y: targetY });
    var target = make("text", { x: targetX, y: targetY });
    var right  = make("text", { y: targetY });
    letters.appendChild(left); letters.appendChild(target); letters.appendChild(right);
    svg.appendChild(letters);

    // the slider and its readout, under the image
    var controls = document.createElement("div");
    controls.className = "demo-controls";
    var slider = document.createElement("input");
    slider.type = "range";
    slider.min = smallest; slider.max = largest; slider.step = 0.01; slider.value = 0.25;
    slider.setAttribute("aria-label", "Flanker spacing");
    var readout = document.createElement("span");
    controls.appendChild(slider);
    controls.appendChild(readout);

    function newLetters() {
      target.textContent = randomLetter([""]);
      left.textContent   = randomLetter([target.textContent]);
      right.textContent  = randomLetter([target.textContent, left.textContent]);
    }

    function placeFlankers() {
      var fraction = Number(slider.value);
      var spacing  = fraction * eccentricity;
      left.setAttribute("x", targetX - spacing);
      right.setAttribute("x", targetX + spacing);
      var where = fraction < 0.5 ? "inside Bouma's window" : "outside Bouma's window";
      readout.textContent = "spacing = " + fraction.toFixed(2) + " × eccentricity (" + where + ")";
    }

    slider.addEventListener("input", placeFlankers);
    svg.addEventListener("click", newLetters);

    newLetters();
    placeFlankers();
    box.appendChild(svg);
    box.appendChild(controls);
  }

  var boxes = document.querySelectorAll('[data-demo="bouma"]');
  for (var i = 0; i < boxes.length; i++) boumaDemo(boxes[i]);
})();
