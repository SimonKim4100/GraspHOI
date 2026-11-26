document.addEventListener('DOMContentLoaded', function () {
  bulmaCarousel.attach('#results-carousel', {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    navigationKeys: false,
    breakpoints: [],  // one example per view at every window width
  });

  document.querySelectorAll('#results-carousel model-viewer').forEach(function (viewer) {
    // Dragging a model rotates it instead of swiping to another slide.
    ['mousedown', 'touchstart'].forEach(function (type) {
      viewer.addEventListener(type, function (event) {
        event.stopPropagation();
      });
    });

    // An empty pan target replaces the default one, a dot at the center of the viewer that blocks dragging.
    var panTarget = document.createElement('div');
    panTarget.slot = 'pan-target';
    viewer.appendChild(panTarget);
  });
});
