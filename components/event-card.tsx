import Image from "next/image"
import Link from "next/link"
import { Card, CardContent, CardFooter } from "@/components/ui/card"

interface EventCardProps {
  event: {
    id: string
    name: string
    description: string
    imageUrl: string
    imageCount: number
  }
}

export function EventCard({ event }: EventCardProps) {
  return (
    <Link href={`/events/${event.id}`}>
      <Card className="overflow-hidden h-full transition-all hover:shadow-md">
        <div className="relative h-48 w-full">
          <Image
            src={event.imageUrl || "/placeholder.svg?height=200&width=400"}
            alt={event.name}
            fill
            className="object-cover"
          />
        </div>
        <CardContent className="p-4">
          <h3 className="text-xl font-semibold mb-2">{event.name}</h3>
          <p className="text-muted-foreground line-clamp-2">{event.description}</p>
        </CardContent>
        <CardFooter className="p-4 pt-0 text-sm text-muted-foreground">
          {event.imageCount} {event.imageCount === 1 ? "image" : "images"}
        </CardFooter>
      </Card>
    </Link>
  )
}

