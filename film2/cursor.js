(() => { const c = document.createElement('div'); c.id = '__cur';
 c.innerHTML = '<svg width="26" height="34" viewBox="0 0 26 34"><path d="M2 2 L2 26 L8.5 20 L13 31 L17.5 29 L13 18.5 L22 18.5 Z" fill="#fff" stroke="#111" stroke-width="2" stroke-linejoin="round"/></svg>';
 Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex:99999, pointerEvents:'none', transform:'translate(-3px,-2px)', filter:'drop-shadow(0 2px 4px rgba(0,0,0,.45))', transition:'none'});
 document.body.appendChild(c); window.__cx = 760; window.__cy = 520; c.style.left = '760px'; c.style.top = '520px';
 window.__move = (x, y, ms) => new Promise(r => { const x0 = window.__cx, y0 = window.__cy, t0 = performance.now();
   const step = now => { const k = Math.min(1, (now - t0) / ms), e = k < .5 ? 4*k*k*k : 1 - Math.pow(-2*k + 2, 3) / 2;
     window.__cx = x0 + (x - x0) * e; window.__cy = y0 + (y - y0) * e; c.style.left = window.__cx + 'px'; c.style.top = window.__cy + 'px';
     k < 1 ? requestAnimationFrame(step) : r(); }; requestAnimationFrame(step); });
 window.__ripple = () => { const d = document.createElement('div'); Object.assign(d.style, {position:'fixed', left:(window.__cx-16)+'px', top:(window.__cy-16)+'px',
   width:'32px', height:'32px', borderRadius:'50%', border:'2px solid rgba(255,255,255,.9)', zIndex:99998, pointerEvents:'none',
   transition:'transform .45s ease-out, opacity .45s ease-out'}); document.body.appendChild(d);
   requestAnimationFrame(() => { d.style.transform = 'scale(1.9)'; d.style.opacity = '0'; }); setTimeout(() => d.remove(), 600); };
})();