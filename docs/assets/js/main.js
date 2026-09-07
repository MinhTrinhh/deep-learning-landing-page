/**
 * Deep Learning & Its Applications (CO3133) - Main Interactive Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  // Tab Switcher Functionality
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  if (tabBtns.length > 0 && tabContents.length > 0) {
    tabBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-tab');

        // Deactivate all tabs
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        // Activate selected tab
        btn.classList.add('active');
        const targetElement = document.getElementById(target);
        if (targetElement) {
          targetElement.classList.add('active');
        }
      });
    });
  }

  // Active Link Highlight based on current path
  const currentPath = window.location.pathname.split('/').pop() || 'index.html';
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const href = link.getAttribute('href');
    if (href && (href.endsWith(currentPath) || (currentPath === '' && href.includes('index.html')))) {
      link.classList.add('active');
    }
  });

  // Copy Code / Disclosure Snippet Button Handler
  const copyBtns = document.querySelectorAll('.btn-copy');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.getAttribute('data-copy-target');
      const codeElem = document.getElementById(targetId);
      if (codeElem) {
        navigator.clipboard.writeText(codeElem.innerText).then(() => {
          const originalText = btn.innerHTML;
          btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied!';
          setTimeout(() => {
            btn.innerHTML = originalText;
          }, 2000);
        });
      }
    });
  });
});
