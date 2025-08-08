import { SpeedInsights } from "@vercel/speed-insights/react"
import './App.css'
import AIBusinessPoster from './components/LandingPage'


function App() {

  return (
    <>
      <AIBusinessPoster /> 
      <SpeedInsights />
    </>
  )
}

export default App
