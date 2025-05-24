import './App.css'
import { GameOfLifeSimple } from './components/GameOfLifeSimple'

function App() {
  return (
    <div className="p-5">
      <div className="max-w-2xl mx-auto flex justify-center">
        <GameOfLifeSimple />
      </div>
    </div>
  )
}

export default App
