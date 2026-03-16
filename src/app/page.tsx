'use client'

import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'

export default function Home() {
  const [data, setData] = useState<{ message: string } | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/hello')
      .then((res) => res.json())
      .then((data) => {
        setData(data)
        setLoading(false)
      })
  }, [])

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <h1 className="text-4xl font-bold">Next.js with Python API</h1>
      <div className="mt-8">
        {loading ? (
          <p>Loading...</p>
        ) : (
          <p className="text-lg">{data?.message}</p>
        )}
      </div>
      <div className="mt-8">
        <Button>Click me</Button>
      </div>
    </main>
  )
}
