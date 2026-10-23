(function () {
  var svg = document.getElementById('pl-svg');
  if (!svg) return;

  var METHODS = {
    lmp: {
      caption: 'DMP weights from GPT-5.2, with up to 3 rounds of human feedback.',
      motion: ['DMP weights', '+ goal offset'],
      per: 'labels, masks, depth',
      gen: 'GPT-5.2',
      stages: 7,
      detectAt: 0.8
    },
    jf: {
      caption: 'Same as LMP, first attempt only. No feedback.',
      motion: ['DMP weights', '+ goal offset'],
      per: 'labels, masks, depth',
      gen: 'GPT-5.2',
      stages: 5,
      detectAt: 0.8
    },
    trajgen: {
      caption: 'No decomposer. GPT-5.2 writes code that calls detect_object() and outputs waypoints.',
      motion: ['Waypoints', 'from generated code'],
      per: 'masks, depth, on request',
      gen: 'GPT-5.2 writes code',
      stages: 5,
      detectAt: 1.8
    },
    pi: {
      caption: 'Raw images and the task instruction. Fine-tuned on 5 demos per task.',
      motion: ['DMP weights', '+ goal offset'],
      per: 'labels, masks, depth',
      gen: 'GPT-5.2',
      stages: 2,
      detectAt: 0
    }
  };
  var ORDER = ['lmp', 'jf', 'trajgen', 'pi'];
  var STAGE_MS = 750;
  var HOLD_MS = 900;
  var LOOPS_PER_METHOD = 2;

  var tabs = document.querySelectorAll('.pl-tab');
  var nodes = svg.querySelectorAll('.pl-node');
  var edges = svg.querySelectorAll('.pl-edge');
  var packets = document.getElementById('pl-packets');
  var perception = document.getElementById('pl-perception');
  var caption = document.getElementById('pl-caption');
  var motionTitle = document.getElementById('pl-motion-title');
  var motionSub = document.getElementById('pl-motion-sub');
  var perSub = document.getElementById('pl-per-sub');
  var genSub = document.getElementById('pl-gen-sub');
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var method = 'lmp';
  var autoplay = !reduceMotion;
  var visible = false;
  var t0 = null;
  var loops = 0;

  // edges carry per-method stages ("lmp:0 trajgen:1"), nodes a method list
  function stageOf(el, m) {
    var spec = el.getAttribute('data-stages');
    if (!spec) return null;
    var hit = spec.split(' ').filter(function (kv) { return kv.split(':')[0] === m; })[0];
    return hit ? +hit.split(':')[1] : null;
  }

  function uses(el, m) {
    if (el.hasAttribute('data-stages')) return stageOf(el, m) !== null;
    return el.getAttribute('data-methods').split(' ').indexOf(m) !== -1;
  }

  function select(m) {
    method = m;
    tabs.forEach(function (tab) {
      tab.classList.toggle('is-active', tab.getAttribute('data-method') === m);
    });
    nodes.forEach(function (n) { n.classList.toggle('off', !uses(n, m)); });
    edges.forEach(function (e) { e.classList.toggle('off', !uses(e, m)); });
    motionTitle.textContent = METHODS[m].motion[0];
    motionSub.textContent = METHODS[m].motion[1];
    perSub.textContent = METHODS[m].per;
    genSub.textContent = METHODS[m].gen;
    caption.textContent = METHODS[m].caption;
    perception.classList.toggle('detect', reduceMotion && uses(perception, m));
    packets.innerHTML = '';
    t0 = null;
    loops = 0;
  }

  function ease(x) {
    return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
  }

  function draw(stage, frac) {
    packets.innerHTML = '';
    edges.forEach(function (e) {
      if (stageOf(e, method) !== stage) return;
      var len = e.getTotalLength();
      var p = e.getPointAtLength(ease(frac) * len);
      var c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('class', 'pl-packet');
      c.setAttribute('cx', p.x);
      c.setAttribute('cy', p.y);
      c.setAttribute('r', 6);
      packets.appendChild(c);
    });
  }

  function tick(now) {
    requestAnimationFrame(tick);
    if (!visible || reduceMotion) return;
    if (t0 === null) t0 = now;
    var cfg = METHODS[method];
    var cycle = cfg.stages * STAGE_MS + HOLD_MS;
    var t = now - t0;

    if (t >= cycle) {
      loops += 1;
      if (autoplay && loops >= LOOPS_PER_METHOD) {
        select(ORDER[(ORDER.indexOf(method) + 1) % ORDER.length]);
      } else {
        t0 = now;
      }
      return;
    }

    var stage = Math.floor(t / STAGE_MS);
    if (stage < cfg.stages) {
      draw(stage, (t % STAGE_MS) / STAGE_MS);
    } else {
      packets.innerHTML = '';
    }
    // perception lights up as the camera packet arrives
    perception.classList.toggle('detect', uses(perception, method) && t > STAGE_MS * cfg.detectAt);
  }

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function () {
      autoplay = false;
      select(tab.getAttribute('data-method'));
    });
  });

  if ('IntersectionObserver' in window) {
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
    }, { threshold: 0.2 }).observe(svg);
  } else {
    visible = true;
  }

  select(method);
  requestAnimationFrame(tick);
})();

document.querySelectorAll('.prompt-copy').forEach(function (btn) {
  btn.addEventListener('click', function () {
    var text = btn.parentNode.querySelector('pre').innerText;
    navigator.clipboard.writeText(text).then(function () {
      btn.textContent = 'Copied';
      setTimeout(function () { btn.textContent = 'Copy'; }, 1200);
    });
  });
});
