function load_item () {
  var length = localStorage.getItem('length');

  for (var i = 0; i < length; i++) {
    var result = localStorage.getItem('result' + i),
        item = localStorage.getItem('item' + i);
    if(!result) {
      localStorage.setItem('current', i);
      window.location = 'https://scholar.google.com/scholar?q="' + item + '"';
      return ;
    }
  }
}
function load_year () {
  var current = localStorage.getItem('current');
  var year = document.querySelector('.gs_a')

	localStorage.setItem('result' + current, year.innerHTML);
}
load_year();
load_item();
