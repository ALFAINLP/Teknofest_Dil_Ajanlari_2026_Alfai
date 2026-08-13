export const APP_CONFIG = Object.freeze({
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8080',
  eventRevealIntervalMs: 650,
  operatorFeedbackStorageKey: 'alfai-operator-feedback',
})
