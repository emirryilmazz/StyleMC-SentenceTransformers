$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    function showResult(file_name){
        var file_path = 'http://127.0.0.1:5000/images?filename=' + file_name
        console.log('path', file_path)
        $('#result').hide()
        $('#result').css('background-image', 'none');
        setTimeout({}, 500)
        $('#result').css('background-image', 'url(' + file_path + ')');
        $('#result').show();
        $('.final-img-preview').show();
        $('#result').fadeIn(650);
    }

    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#result').hide();
        readURL(this);
    });
    // Predict
    $('#btn-predict').click(function () {
        var form = document.getElementById('upload-file');
        var form_data = new FormData(form);
        $('#result').css('background-image', 'url("")');
        $('#result').css('background-image', 'none');

        // Show loading animation
        $('.loader').show();
        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                showResult(data)
                console.log('Success!');
            },
        });
    });

});
