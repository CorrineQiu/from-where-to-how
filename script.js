const revealItems = document.querySelectorAll(".reveal");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("is-visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.1, rootMargin: "0px 0px -50px" },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
}

const backToTop = document.querySelector(".back-to-top");
const setBackToTopVisibility = () => {
  backToTop.classList.toggle("is-visible", window.scrollY > 620);
};

window.addEventListener("scroll", setBackToTopVisibility, { passive: true });
setBackToTopVisibility();
backToTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" }));

const lightbox = document.querySelector("#lightbox");
const lightboxImage = document.querySelector("#lightbox-image");
const lightboxCaption = document.querySelector("#lightbox-caption");
const lightboxClose = document.querySelector(".lightbox-close");

const openLightbox = (sourceImage) => {
  lightboxImage.src = sourceImage.currentSrc || sourceImage.src;
  lightboxImage.alt = sourceImage.alt;
  const parentFigure = sourceImage.closest("figure");
  const figureCaption = parentFigure ? parentFigure.querySelector("figcaption") : null;
  lightboxCaption.textContent = figureCaption ? figureCaption.textContent.trim() : sourceImage.alt;
  lightbox.showModal();
};

document.querySelectorAll("img[data-lightbox]").forEach((image) => {
  image.tabIndex = 0;
  image.setAttribute("role", "button");
  image.setAttribute("aria-label", `${image.alt} Open larger view.`);
  image.addEventListener("click", () => openLightbox(image));
  image.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLightbox(image);
    }
  });
});

lightboxClose.addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

document.querySelector("#current-year").textContent = new Date().getFullYear();
