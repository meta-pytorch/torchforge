// Lightbox functionality for Mermaid diagrams
document.addEventListener('DOMContentLoaded', function() {
    // Create lightbox modal for Mermaid diagrams
    const lightbox = document.createElement('div');
    lightbox.id = 'mermaid-lightbox';
    lightbox.style.cssText = `
        display: none;
        position: fixed;
        z-index: 9999;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.9);
        cursor: zoom-out;
    `;

    const lightboxContent = document.createElement('div');
    lightboxContent.style.cssText = `
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 95%;
        max-height: 95%;
        overflow: auto;
    `;

    const closeBtn = document.createElement('span');
    closeBtn.innerHTML = '&times;';
    closeBtn.style.cssText = `
        position: absolute;
        top: 15px;
        right: 35px;
        color: #f1f1f1;
        font-size: 40px;
        font-weight: bold;
        cursor: pointer;
        z-index: 10000;
    `;
    closeBtn.onclick = function() {
        lightbox.style.display = 'none';
    };

    lightbox.appendChild(closeBtn);
    lightbox.appendChild(lightboxContent);
    document.body.appendChild(lightbox);

    // Click outside to close
    lightbox.onclick = function(e) {
        if (e.target === lightbox) {
            lightbox.style.display = 'none';
        }
    };

    // ESC key to close
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox.style.display === 'block') {
            lightbox.style.display = 'none';
        }
    });

    // Make all Mermaid diagrams clickable
    const mermaidDivs = document.querySelectorAll('.mermaid');
    mermaidDivs.forEach(function(div) {
        div.style.cursor = 'zoom-in';
        div.title = 'Click to enlarge';

        div.addEventListener('click', function() {
            const clone = div.cloneNode(true);

            // Style the cloned diagram to fill the lightbox
            clone.style.cssText = `
                cursor: default;
                width: 90vw;
                max-width: 1400px;
                height: auto;
                margin: auto;
            `;

            // Find and resize the SVG inside
            const svg = clone.querySelector('svg');
            if (svg) {
                svg.style.cssText = `
                    width: 100% !important;
                    height: auto !important;
                    max-width: none !important;
                    max-height: 90vh !important;
                `;
                svg.removeAttribute('width');
                svg.removeAttribute('height');
            }

            lightboxContent.innerHTML = '';
            lightboxContent.appendChild(clone);
            lightbox.style.display = 'block';
        });
    });
});

// Custom JavaScript to make long parameter lists in class signatures collapsible
document.addEventListener('DOMContentLoaded', function() {
    console.log('Collapsible parameters script loaded');

    // Find all class/function signatures
    const signatures = document.querySelectorAll('dl.py.class > dt, dl.py.function > dt, dl.py.method > dt');

    signatures.forEach(function(signature) {
        // Find all parameter elements in the signature
        const params = signature.querySelectorAll('em.sig-param, .sig-param');

        console.log(`Found signature with ${params.length} parameters`);

        // Only make it collapsible if there are more than 10 parameters
        if (params.length > 10) {
            console.log('Creating collapsible structure for signature with', params.length, 'parameters');

            const visibleCount = 5;
            const hiddenCount = params.length - visibleCount;

            // Create a wrapper div for the toggle button
            const wrapper = document.createElement('span');
            wrapper.className = 'sig-params-wrapper';
            wrapper.style.display = 'inline';

            // Create toggle button
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'params-toggle-btn-inline';
            toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-right"></i> Show More`;
            toggleBtn.setAttribute('aria-expanded', 'false');
            toggleBtn.title = `Show ${hiddenCount} more parameters`;

            // Collect all nodes to hide (params and text nodes between them)
            const nodesToHide = [];

            // Hide parameters after the first 3
            let insertedButton = false;
            params.forEach(function(param, index) {
                if (index >= visibleCount) {
                    // Add 'hidden' class to hide the parameter
                    param.classList.add('sig-param-hidden');
                    nodesToHide.push(param);

                    // Also hide the text node (comma/space) that follows this parameter
                    let nextNode = param.nextSibling;
                    while (nextNode && nextNode.nodeType === Node.TEXT_NODE) {
                        const textSpan = document.createElement('span');
                        textSpan.className = 'sig-param-hidden';
                        textSpan.textContent = nextNode.textContent;
                        nextNode.parentNode.replaceChild(textSpan, nextNode);
                        nodesToHide.push(textSpan);
                        break;
                    }

                    // Insert the toggle button before the first hidden parameter
                    if (!insertedButton) {
                        param.parentNode.insertBefore(wrapper, param);
                        wrapper.appendChild(toggleBtn);
                        insertedButton = true;
                    }
                }
            });

            // Add click handler to toggle
            toggleBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();

                const isExpanded = toggleBtn.getAttribute('aria-expanded') === 'true';

                if (isExpanded) {
                    // Collapse: hide parameters again
                    nodesToHide.forEach(function(node) {
                        node.classList.add('sig-param-hidden');
                    });
                    toggleBtn.setAttribute('aria-expanded', 'false');
                    toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-right"></i> Show More`;
                    toggleBtn.title = `Show ${hiddenCount} more parameters`;
                } else {
                    // Expand: show all parameters
                    nodesToHide.forEach(function(node) {
                        node.classList.remove('sig-param-hidden');
                    });
                    toggleBtn.setAttribute('aria-expanded', 'true');
                    toggleBtn.innerHTML = `<i class="fa-solid fa-chevron-down"></i> Hide`;
                    toggleBtn.title = `Hide ${hiddenCount} parameters`;
                }
            });

            console.log('Collapsible structure created successfully');
        }
    });
});
