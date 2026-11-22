(function () {
  'use strict';

  var storageKey = 'site-language';
  var root = document.documentElement;
  var translationEl = document.getElementById('site-translations');
  var dictionary = translationEl ? JSON.parse(translationEl.textContent) : {};
  var title = document.querySelector('title');
  var titleKey = title ? title.getAttribute('data-i18n-title') : null;
  var titleSuffix = title ? title.getAttribute('data-i18n-suffix') : '';

  function translate(key, language) {
    return Object.prototype.hasOwnProperty.call(dictionary, key)
      ? dictionary[key][language] : null;
  }

  function updateComments(language) {
    var loader = document.querySelector('script[src="https://giscus.app/client.js"]');
    if (loader) loader.setAttribute('data-lang', language);
    var frame = document.querySelector('iframe.giscus-frame');
    if (frame && frame.contentWindow) {
      frame.contentWindow.postMessage({giscus: {setConfig: {lang: language}}}, 'https://giscus.app');
    }
  }

  function applyLanguage(language, persist) {
    if (language !== 'zh-CN' && language !== 'en') return;
    root.lang = language;
    document.querySelectorAll('[data-i18n]').forEach(function (element) {
      var value = translate(element.getAttribute('data-i18n'), language);
      if (value != null) element.textContent = value;
    });
    document.querySelectorAll('[data-i18n-aria-label]').forEach(function (element) {
      var value = translate(element.getAttribute('data-i18n-aria-label'), language);
      if (value != null) element.setAttribute('aria-label', value);
    });
    if (title && titleKey) {
      var translatedTitle = translate(titleKey, language);
      if (translatedTitle != null) title.textContent = translatedTitle + titleSuffix;
    }
    updateComments(language);
    document.querySelectorAll('.language-switcher').forEach(function (element) {
      element.hidden = false;
      var select = element.querySelector('select');
      if (select) select.value = language;
      element.querySelectorAll('.lang-btn[data-lang]').forEach(function (btn) {
        var active = btn.getAttribute('data-lang') === language;
        btn.classList.toggle('is-active', active);
        btn.setAttribute('aria-pressed', active ? 'true' : 'false');
      });
    });
    if (persist) {
      try { localStorage.setItem(storageKey, language); } catch (error) {
        // Language switching still works when browser storage is unavailable.
      }
    }
  }

  function initialize() {
    document.addEventListener('load', function (event) {
      if (event.target.matches && event.target.matches('iframe.giscus-frame')) {
        updateComments(root.lang);
      }
    }, true);
    applyLanguage(root.lang, false);
    document.querySelectorAll('.language-switcher select').forEach(function (select) {
      select.addEventListener('change', function () {
        applyLanguage(select.value, true);
      });
    });
    document.querySelectorAll('.language-switcher .lang-btn[data-lang]').forEach(function (btn) {
      btn.addEventListener('click', function () {
        applyLanguage(btn.getAttribute('data-lang'), true);
      });
    });
    window.addEventListener('storage', function (event) {
      if (event.key === storageKey) applyLanguage(event.newValue, false);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }
}());
