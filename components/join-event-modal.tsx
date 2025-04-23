import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { joinEvent } from '@/lib/api';

interface JoinEventModalProps {
  userId: string;
}

export function JoinEventModal({ userId }: JoinEventModalProps) {
  async function handleSubmit(formData: FormData) {

    const eventCode = formData.get('eventCode') as string;

    try {
      const result = await joinEvent(eventCode);

      if (result.error) {
        return { error: result.error };
      }

      return { success: true };
    } catch (err) {
      return { error: 'Failed to join event. Please try again.' };
    }
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button>Join Event</Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Join Event</DialogTitle>
          <DialogDescription>Enter the event code to join an event.</DialogDescription>
        </DialogHeader>
        <form action={handleSubmit}>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="event-code">Event Code</Label>
              <Input
                id="event-code"
                name="eventCode"
                placeholder="Enter event code"
                required
              />
            </div>
          </div>
          <DialogFooter>
            <Button type="submit">Join Event</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}