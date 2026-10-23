// Task chips for the gripper-path comparison.
(function () {
  var block = document.querySelector('.paths-block');
  if (!block) return;
  var chips = block.querySelectorAll('.dp-chip');
  var imgs = block.querySelectorAll('.path-img');
  var caption = block.querySelector('.paths-caption');
  chips.forEach(function (chip) {
    chip.addEventListener('click', function () {
      var task = chip.getAttribute('data-task');
      chips.forEach(function (c) { c.classList.toggle('is-active', c === chip); });
      imgs.forEach(function (img) {
        img.src = './static/resources/paths/path_' + task + '_' + img.getAttribute('data-method') + '.jpg';
      });
      caption.textContent = chip.getAttribute('data-caption');
    });
  });
})();
