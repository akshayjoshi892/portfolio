import 'dotenv/config'
import bcrypt from 'bcryptjs'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  const email = process.env.SEED_ADMIN_EMAIL ?? 'admin@example.com'
  const password = process.env.SEED_ADMIN_PASSWORD ?? 'change-me-please'
  const name = process.env.SEED_ADMIN_NAME ?? 'Admin'

  const passwordHash = await bcrypt.hash(password, 12)

  await prisma.user.upsert({
    where: { email },
    create: { email, passwordHash, name },
    update: { passwordHash, name },
  })

  console.log(`Seeded admin user: ${email}`)
}

main()
  .catch((err) => {
    console.error(err)
    process.exit(1)
  })
  .finally(() => prisma.$disconnect())
