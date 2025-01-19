"use client"
import React from 'react'
import Dashboard from '../page';
import { LayoutDashboard, LeafyGreenIcon, UserCircle } from 'lucide-react';
import { usePathname } from 'next/navigation';
import Link from 'next/link';


function SideBar() {

    const MenuList = [
        {
            name: 'Dashboard',
            icon: LayoutDashboard,
            path: '/dashboard',
        },
        {
            name: 'MathAi',
            icon: LeafyGreenIcon,
            path: 'http://localhost:5173/',
        },
        {
            name: 'Profile',
            icon: UserCircle,
            path: '/dashboard/profile',        }
    ]

    const path = usePathname();

  return (
    <div className="h-screen shadow-md p-5">
      <div className="flex gap-2 items-center">
        <img src="/logo.svg" alt="logo" width={40} height={40} />
        <h2 className="font-bold text-2xl">TutorAi</h2>
      </div>
      <div className="mt-10">
        <Link href='/create' className='w-full'>
            <button className="btn btn-outline-primary w-full">+ Create New</button>
        </Link>

        <div className='mt-5'>
          {MenuList.map((menu, index) => (
          <Link key={index} href={menu.path}>
            <div className={`flex gap-5 items-center p-3 hover:bg-gray-300 rounded-lg cursor-pointer mt-3 ${path==menu.path&&'bg-gray-300'}`}>
                <menu.icon />
                <h2>{menu.name}</h2>
            </div>
          </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

export default SideBar