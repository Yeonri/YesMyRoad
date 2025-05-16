import { FlatCompat } from '@eslint/eslintrc'
import { dirname } from 'path'
import { fileURLToPath } from 'url'

// Plugins
import eslintPluginPrettier from 'eslint-plugin-prettier'
import eslintPluginReact from 'eslint-plugin-react'
import eslintPluginReactHooks from 'eslint-plugin-react-hooks'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)

const compat = new FlatCompat({ baseDirectory: __dirname })

const eslintConfig = [
  {
    ignores: [
      'node_modules',
      'build',
      'dist',
      'package-lock.json',
      '*.config.js',
      '*.config.cjs',
      '*.config.mjs',
    ],
    files: ['**/*.ts', '**/*.tsx', '**/*.js', '**/*.jsx'],
  },
  ...compat.extends(
    // next
    'next/core-web-vitals',
    'next/typescript',

    // prettier
    'plugin:prettier/recommended',

    // react
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ),
  {
    plugins: {
      // import
      'eslint-plugin-react': eslintPluginReact,
      'eslint-plugin-react-hooks': eslintPluginReactHooks,
      'eslint-plugin-prettier': eslintPluginPrettier,
    },
    rules: {
      // react
      'react/react-in-jsx-scope': 'off', // Next.js에서 필요 없음
      'react/prop-types': 'off', // TypeScript에서 필요 없음

      // react hooks
      'react-hooks/rules-of-hooks': 'error', // 훅 규칙 검사
      'react-hooks/exhaustive-deps': 'warn', // 의존성 배열 검사

      // typescript
      '@typescript-eslint/no-unused-vars': 'warn', // 사용하지 않는 변수 설정
      '@typescript-eslint/no-explicit-any': 'warn', // any 타입 설정
    },
  },
]

export default eslintConfig
