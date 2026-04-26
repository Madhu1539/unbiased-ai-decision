import { useState, useEffect } from 'react'

const STORAGE_KEY = 'unbiased-ai-theme'

/**
 * Returns the initial theme:
 *   1. localStorage saved preference  (highest priority)
 *   2. OS/browser prefers-color-scheme
 *   3. Fallback → 'dark'
 */
function resolveInitialTheme() {
  const saved = localStorage.getItem(STORAGE_KEY)
  if (saved === 'dark' || saved === 'light') return saved
  if (window.matchMedia?.('(prefers-color-scheme: light)').matches) return 'light'
  return 'dark'
}

export function useTheme() {
  const [theme, setTheme] = useState(resolveInitialTheme)

  // Apply / remove 'dark' class on <html> and persist preference
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem(STORAGE_KEY, theme)
  }, [theme])

  const toggleTheme = () =>
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'))

  return { theme, toggleTheme }
}
