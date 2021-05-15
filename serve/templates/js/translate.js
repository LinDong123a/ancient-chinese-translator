$("#submit").click(() =>　{
    var text = $("#text").val();
    $.ajax({
        type:'post',
        contentType: 'application/json',
        url: '/query/',
        data: JSON.stringify({text: text}),
        success: function(data) {
            console.log(data);

            $("#ancient").val(data["text"]);
        },
        error: function(error) {
            console.log(error);
            alert(error.responseJSON.msg)
        }
    })
})
