import { useState } from 'react'
import './App.css'
import AIBusinessPoster from './components/LandingPage'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
      <AIBusinessPoster />
    </>
  )
}

export default App
