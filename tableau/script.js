// Pop-in animation on scroll (re-animate every time)
const popIns = document.querySelectorAll(".pop-in");

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
    } else {
      entry.target.classList.remove("visible");
    }
  });
}, {
  threshold: 0.15
});

popIns.forEach((el) => observer.observe(el));


