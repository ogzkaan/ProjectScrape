/*!
    * Start Bootstrap - SB Admin v7.0.7 (https://startbootstrap.com/template/sb-admin)
    * Copyright 2013-2023 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
    // 
// Scripts
// 

window.addEventListener('DOMContentLoaded', event => {

    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }
    

});

function toggle(element, sublistId) {
    var sublist = document.getElementById(sublistId);
    var isExpanded = sublist.style.display === 'block';
    sublist.style.display = isExpanded ? 'none' : 'block';
    element.classList.toggle('expanded', !isExpanded);
}
function updateDatabase(checkbox, checkbox_id) {
    var isChecked = checkbox.checked ? 'True' : 'False';
    
    fetch(`/product/update/${checkbox_id}/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            // Include CSRF token as header if not exempt
        },
        body: `is_checked=${isChecked}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log(`Product ${data.id} checked state is now ${data.is_checked}`);
        } else {
            console.error('An error occurred:', data.error);
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}