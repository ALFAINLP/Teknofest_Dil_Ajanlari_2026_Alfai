import { useState } from 'react'
import { APP_CONFIG } from '@/config/appConfig'

function readStoredFeedback() {
  try {
    return JSON.parse(localStorage.getItem(APP_CONFIG.operatorFeedbackStorageKey) || '{}')
  } catch {
    return {}
  }
}

export function useOperatorFeedback() {
  const [feedbackMap, setFeedbackMap] = useState(readStoredFeedback)

  function saveFeedback(event, feedback) {
    setFeedbackMap(current => {
      const next = { ...current, [event.id]: feedback }
      localStorage.setItem(APP_CONFIG.operatorFeedbackStorageKey, JSON.stringify(next))
      return next
    })
  }

  function clearFeedback() {
    localStorage.removeItem(APP_CONFIG.operatorFeedbackStorageKey)
    setFeedbackMap({})
  }

  return { feedbackMap, saveFeedback, clearFeedback }
}
