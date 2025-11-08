/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'tars-primary': '#3b82f6',
        'tars-secondary': '#8b5cf6',
        'tars-accent': '#10b981',
        'tars-dark': '#1e293b',
        'tars-darker': '#0f172a',
      },
    },
  },
  plugins: [],
}
