import React from 'react'
import { Sun, Moon } from 'lucide-react'
import { useThemeContext } from '../context/ThemeContext'

/**
 * Animated pill-style toggle button.
 * - Sun icon  → Light Mode active
 * - Moon icon → Dark Mode active
 * - Tooltip shows next action  ("Switch to Dark Mode" / "Switch to Light Mode")
 * - Full keyboard accessible (button element)
 */
export default function ThemeToggle() {
  const { theme, toggleTheme } = useThemeContext()
  const isDark = theme === 'dark'

  return (
    <button
      id="theme-toggle"
      onClick={toggleTheme}
      aria-label={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      className={`
        relative group flex items-center gap-1.5 px-3 py-1.5 rounded-xl
        border text-xs font-semibold
        transition-all duration-300 ease-in-out
        focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-400 focus-visible:ring-offset-2
        ${isDark
          ? 'bg-surface-700 border-surface-600 text-slate-300 hover:bg-surface-600 hover:text-white focus-visible:ring-offset-surface-900'
          : 'bg-amber-50 border-amber-200 text-amber-700 hover:bg-amber-100 focus-visible:ring-offset-white'
        }
      `}
    >
      {/* Animated icon */}
      <span
        className="transition-transform duration-300"
        style={{ transform: isDark ? 'rotate(0deg)' : 'rotate(180deg)' }}
      >
        {isDark
          ? <Moon size={14} className="text-brand-400" />
          : <Sun size={14} className="text-amber-500" />
        }
      </span>

      {/* Label */}
      <span>{isDark ? 'Dark' : 'Light'}</span>

      {/* Tooltip */}
      <span
        role="tooltip"
        className={`
          pointer-events-none absolute -bottom-9 left-1/2 -translate-x-1/2
          whitespace-nowrap px-2 py-1 rounded-lg text-[10px] font-medium shadow-lg
          opacity-0 group-hover:opacity-100 transition-opacity duration-200
          ${isDark ? 'bg-surface-700 text-slate-200' : 'bg-gray-800 text-white'}
        `}
      >
        {isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
      </span>
    </button>
  )
}
