import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-inter)', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        brand: {
          green: '#76B900',
          'green-dim': '#5a8c00',
          'green-muted': '#3d6000',
        },
        surface: {
          DEFAULT: '#16181c',
          elevated: '#1d2026',
          overlay: '#24272f',
        },
        border: {
          DEFAULT: '#1f2229',
          strong: '#2a2d35',
          stronger: '#3a3d47',
        },
      },
    },
  },
  plugins: [],
}

export default config
