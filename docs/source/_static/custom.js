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
