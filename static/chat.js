 // Get references to the slider elements and their corresponding output elements
  // Get references to the slider elements and their corresponding output elements
  const slider1 = document.getElementById('slider1');
  const output1 = document.querySelector('output[for="slider1"]');
  const slider2 = document.getElementById('slider2');
  const output2 = document.querySelector('output[for="slider2"]');

  // Set initial values to 0
  output1.textContent = 0
  output2.textContent = 0

  // Add event listeners to update the output when sliders are moved
  slider1.addEventListener('input', function () {
      output1.textContent = slider1.value;
  });

  slider2.addEventListener('input', function () {
      output2.textContent = slider2.value;
  });


  function toggleOutputData() {
    var outputData = document.getElementById("output-data");
    if (outputData.style.display === "none" || outputData.style.display === "") {
      outputData.style.display = "block";
    } else {
      outputData.style.display = "none";
    }
  }
// // Use JavaScript to handle form submission
// document.getElementById('userForm').addEventListener('submit', function (e) {
//     e.preventDefault();  // Prevent the default form submission behavior

//     // Get form data
//     var formData = new FormData(this);

//     // Send a POST request to the server using AJAX
//     fetch('/inputs', {
//         method: 'POST',
//         body: formData
//     })
//     .then(response => response.text())
//     .then(data => {
//         // Update the 'result' div with the response from the server
//         document.getElementById('result').innerHTML = data;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });



