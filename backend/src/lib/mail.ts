interface ContactPayload {
  name: string
  email: string
  message: string
}

export async function sendContactEmail({ name, email, message }: ContactPayload) {
  const apiKey = process.env.RESEND_API_KEY
  const to = process.env.CONTACT_TO_EMAIL
  if (!apiKey || !to) {
    console.log('[mail] RESEND_API_KEY/CONTACT_TO_EMAIL not set — skipping email send.')
    return
  }

  await fetch('https://api.resend.com/emails', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      from: 'Portfolio Contact <contact@yourdomain.com>',
      to,
      reply_to: email,
      subject: `New message from ${name}`,
      text: message,
    }),
  })
}
