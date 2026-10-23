// Tabs under the overview. The URL hash picks the tab, so #playground or #prompts links still work.
(function () {
  var nav = document.getElementById('subnav');
  if (!nav) return;
  var tabs = Array.prototype.slice.call(nav.querySelectorAll('a[data-page]'));
  var pages = Array.prototype.slice.call(document.querySelectorAll('.subpage'));

  function pageOf(id) {
    var el = id && document.getElementById(id);
    var page = el && el.closest('.subpage');
    return page ? page.getAttribute('data-page') : null;
  }

  function show(name) {
    pages.forEach(function (p) { p.hidden = p.getAttribute('data-page') !== name; });
    tabs.forEach(function (t) { t.classList.toggle('is-active', t.getAttribute('data-page') === name); });
  }

  function route(scroll) {
    var id = decodeURIComponent(location.hash.slice(1));
    var name = pageOf(id) || tabs[0].getAttribute('data-page');
    show(name);
    if (!scroll || !id) return;
    // tab names point at the tab bar; deeper anchors scroll to themselves
    var target = tabs.some(function (t) { return t.getAttribute('data-page') === id; }) ? nav : document.getElementById(id);
    if (target) target.scrollIntoView();
  }

  tabs.forEach(function (tab) {
    tab.addEventListener('click', function (e) {
      e.preventDefault();
      var name = tab.getAttribute('data-page');
      if (location.hash !== '#' + name) history.pushState(null, '', '#' + name);
      show(name);
      if (nav.getBoundingClientRect().top < 0) nav.scrollIntoView();
    });
  });

  window.addEventListener('popstate', function () { route(true); });
  window.addEventListener('hashchange', function () { route(true); });
  route(true);
})();
