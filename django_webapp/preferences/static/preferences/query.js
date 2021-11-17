document.addEventListener('keydown', (e) => {
    if (e.code == 'ArrowLeft') {
        document.getElementById('leftButton').click()
    }
    else if (e.code == 'ArrowUp') {
        document.getElementById('equalsButton').click()
    }
    else if (e.code == 'ArrowDown') {
        document.getElementById('skipButton').click()
    }
    else if (e.code == 'ArrowRight') {
        document.getElementById('rightButton').click()
    }
    else {
        //do nothing
    }
})