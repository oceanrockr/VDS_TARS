# UI FRAMEWORK — T.A.R.S. LocalLLM Desktop

**Project:** T.A.R.S. (Temporal Augmented Retrieval System)  
**Primary Platform:** Windows 11 Desktop  
**Design Philosophy:** Modern, performant, accessible

## Technology Stack

### Core Framework
- **Desktop Shell:** Electron 28+ (cross-platform native wrapper)
- **Frontend Framework:** React 18 with TypeScript 5
- **Build Tool:** Vite 5 (lightning-fast HMR, optimized production builds)
- **Styling:** Tailwind CSS 3.4+ (utility-first CSS framework)
- **Component Library:** ShadCN/UI (accessible, customizable primitives)
- **Icons:** Lucide React (consistent, tree-shakeable icon set)

### State Management & Data Fetching
- **Local State:** Zustand (minimal, performant state management)
- **Server State:** React Query / TanStack Query v5 (caching, sync, updates)
- **WebSocket:** Native WebSocket API with custom reconnection wrapper
- **Persistence:** IndexedDB via Dexie.js (structured client-side storage)
- **Forms:** React Hook Form + Zod (type-safe validation)

### UI Enhancement Libraries
- **Animation:** Framer Motion 10+ (declarative animations)
- **Charts:** Recharts 2.x (responsive, composable data visualization)
- **Markdown:** react-markdown + remark-gfm (GitHub Flavored Markdown)
- **Code Highlighting:** react-syntax-highlighter with Prism.js
- **Toast Notifications:** Sonner (lightweight, accessible toasts)
- **Tables:** TanStack Table v8 (headless, powerful data grids)
- **Date/Time:** date-fns (lightweight date utilities, no moment.js)

### Build Targets
- **Primary:** Windows 11 (64-bit) desktop application
- **Secondary:** Windows 11 on Surface Pro (touch-optimized interface)
- **Future:** Linux (Ubuntu 22.04+), Android via Capacitor

---

## Design System

### Color Palette

#### Dark Mode (Default Theme)
```css
/* Background */
--background: 222.2 84% 4.9%         /* slate-950 */
--foreground: 210 40% 98%            /* slate-50 */

/* Cards & Surfaces */
--card: 222.2 84% 4.9%               /* slate-950 */
--card-foreground: 210 40% 98%       /* slate-50 */

/* Primary Actions */
--primary: 199 89% 48%               /* sky-500 - main accent */
--primary-foreground: 210 40% 98%    /* slate-50 */

/* Secondary Elements */
--secondary: 217.2 32.6% 17.5%       /* slate-800 */
--secondary-foreground: 210 40% 98%  /* slate-50 */

/* Accent Highlights */
--accent: 160 84% 39%                /* emerald-500 - success, active states */
--accent-foreground: 210 40% 98%     /* slate-50 */

/* Destructive Actions */
--destructive: 0 84% 60%             /* red-500 */
--destructive-foreground: 210 40% 98% /* slate-50 */

/* Muted Elements */
--muted: 217.2 32.6% 17.5%           /* slate-800 */
--muted-foreground: 215 20.2% 65.1%  /* slate-400 */

/* Borders & Inputs */
--border: 217.2 32.6% 17.5%          /* slate-800 */
--input: 217.2 32.6% 17.5%           /* slate-800 */
--ring: 199 89% 48%                  /* sky-500 - focus rings */

/* Radius */
--radius: 0.5rem                     /* 8px default */
```

#### Light Mode (Optional)
```css
/* Background */
--background: 0 0% 100%              /* white */
--foreground: 222.2 84% 4.9%         /* slate-950 */

/* Cards & Surfaces */
--card: 0 0% 100%                    /* white */
--card-foreground: 222.2 84% 4.9%    /* slate-950 */

/* Primary Actions */
--primary: 199 89% 48%               /* sky-500 */
--primary-foreground: 210 40% 98%    /* slate-50 */

/* Secondary Elements */
--secondary: 210 40% 96.1%           /* slate-100 */
--secondary-foreground: 222.2 47.4% 11.2% /* slate-900 */

/* Accent Highlights */
--accent: 160 84% 39%                /* emerald-500 */
--accent-foreground: 222.2 47.4% 11.2% /* slate-900 */

/* Destructive Actions */
--destructive: 0 84% 60%             /* red-500 */
--destructive-foreground: 210 40% 98% /* slate-50 */

/* Muted Elements */
--muted: 210 40% 96.1%               /* slate-100 */
--muted-foreground: 215.4 16.3% 46.9% /* slate-500 */

/* Borders & Inputs */
--border: 214.3 31.8% 91.4%          /* slate-200 */
--input: 214.3 31.8% 91.4%           /* slate-200 */
--ring: 199 89% 48%                  /* sky-500 */
```

### Typography

**Font Families:**
```css
--font-sans: 'Inter Variable', system-ui, -apple-system, sans-serif
--font-serif: 'DM Serif Display', Georgia, serif
--font-mono: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace
```

**Type Scale (Tailwind Classes):**
```
text-xs:   0.75rem  (12px)  - Labels, captions
text-sm:   0.875rem (14px)  - Small body text
text-base: 1rem     (16px)  - Default body text
text-lg:   1.125rem (18px)  - Emphasized text
text-xl:   1.25rem  (20px)  - Small headings
text-2xl:  1.5rem   (24px)  - Section headings
text-3xl:  1.875rem (30px)  - Page headings
text-4xl:  2.25rem  (36px)  - Hero headings
```

**Font Weights:**
```
font-normal:   400  - Body text
font-medium:   500  - Emphasis, labels
font-semibold: 600  - Subheadings, buttons
font-bold:     700  - Headings
```

**Line Heights:**
```
leading-tight:   1.25  - Headings, condensed text
leading-normal:  1.5   - Default body text
leading-relaxed: 1.75  - Long-form content
```

### Spacing System

Based on 4px grid (Tailwind spacing scale):
```
space-1:  0.25rem (4px)
space-2:  0.5rem  (8px)
space-3:  0.75rem (12px)
space-4:  1rem    (16px)   ← Base unit
space-5:  1.25rem (20px)
space-6:  1.5rem  (24px)
space-8:  2rem    (32px)
space-10: 2.5rem  (40px)
space-12: 3rem    (48px)
space-16: 4rem    (64px)
space-20: 5rem    (80px)
space-24: 6rem    (96px)
```

### Border Radius
```css
rounded-sm:   0.25rem (4px)   - Small elements
rounded-md:   0.5rem  (8px)   - Default buttons, inputs
rounded-lg:   0.75rem (12px)  - Cards, panels
rounded-xl:   1rem    (16px)  - Modal dialogs
rounded-2xl:  1.5rem  (24px)  - Large containers
rounded-full: 9999px          - Pills, avatars
```

### Shadows (Elevation)
```css
shadow-sm:   0 1px 2px 0 rgb(0 0 0 / 0.05)
shadow-md:   0 4px 6px -1px rgb(0 0 0 / 0.1)
shadow-lg:   0 10px 15px -3px rgb(0 0 0 / 0.1)
shadow-xl:   0 20px 25px -5px rgb(0 0 0 / 0.1)
shadow-2xl:  0 25px 50px -12px rgb(0 0 0 / 0.25)
```

---

## Application Layout

### Main Window Structure
```
┌─────────────────────────────────────────────────────┐
│  Title Bar (Electron Frame) - T.A.R.S.             │
│  [Min] [Max] [Close]                      [Settings]│
├──────────┬──────────────────────────────────────────┤
│          │                                          │
│ Sidebar  │        Main Content Area                 │
│ (200px)  │                                          │
│          │   • Chat View (default)                  │
│ Nav      │   • Dashboard                            │
│ Items    │   • Documents                            │
│          │   • Settings                             │
│          │                                          │
│          │                                          │
│          │                                          │
├──────────┴──────────────────────────────────────────┤
│  Status Bar (Optional)                              │
│  Connected • GPU: 45% • 1,247 documents indexed     │
└─────────────────────────────────────────────────────┘
```

### Sidebar Navigation
```
┌────────────────┐
│ [T.A.R.S. Logo]│
│                │
├────────────────┤
│ 💬 Chat        │ ← Active indicator (border-l-4)
│ 📊 Dashboard   │
│ 📁 Documents   │
│ ⚙️ Settings    │
│                │
├────────────────┤
│ 🔌 Online      │ ← Connection status
│ 👤 User        │
└────────────────┘
```

### Responsive Breakpoints
```
mobile:    375px  - 639px   (future web view)
tablet:    640px  - 1023px  (Surface Pro portrait)
desktop:   1024px - 1919px  (primary target)
xl:        1920px - 2559px  (large desktop)
2xl:       2560px+          (4K, multi-monitor)
```

---

## Core UI Components

### 1. Chat View

**Layout Structure:**
```tsx
<div className="flex flex-col h-screen">
  {/* Header */}
  <ChatHeader />
  
  {/* Message Area (scrollable) */}
  <div className="flex-1 overflow-y-auto">
    <MessageList messages={messages} />
  </div>
  
  {/* Input Area (fixed bottom) */}
  <ChatInput onSend={handleSend} />
</div>
```

**Message Bubble Component:**
```tsx
<div className={cn(
  "flex gap-3 px-4 py-6",
  isUser ? "justify-end" : "justify-start"
)}>
  {!isUser && <Avatar>AI</Avatar>}
  
  <div className={cn(
    "max-w-[70%] rounded-lg px-4 py-3",
    isUser 
      ? "bg-primary text-primary-foreground" 
      : "bg-secondary text-secondary-foreground"
  )}>
    <ReactMarkdown>{content}</ReactMarkdown>
    
    {sources && (
      <div className="mt-2 text-xs opacity-70">
        Sources: {sources.join(", ")}
      </div>
    )}
    
    <span className="text-xs opacity-50 mt-1 block">
      {timestamp}
    </span>
  </div>
  
  {isUser && <Avatar>You</Avatar>}
</div>
```

**Typing Indicator:**
```tsx
<div className="flex items-center gap-1 text-muted-foreground px-4">
  <span className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.3s]" />
  <span className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.15s]" />
  <span className="w-2 h-2 bg-current rounded-full animate-bounce" />
  <span className="ml-2">T.A.R.S. is thinking...</span>
</div>
```

**Code Block Rendering:**
```tsx
<div className="relative group">
  <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100">
    <Button size="sm" variant="ghost" onClick={copyCode}>
      <Copy className="h-4 w-4" />
    </Button>
  </div>
  
  <pre className="bg-slate-900 rounded-md p-4 overflow-x-auto">
    <code className="text-sm">
      <SyntaxHighlighter language={language}>
        {code}
      </SyntaxHighlighter>
    </code>
  </pre>
</div>
```

### 2. Admin Dashboard

**Grid Layout:**
```tsx
<div className="p-6 space-y-6">
  {/* Metrics Cards Row */}
  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
    <MetricCard
      title="GPU Load"
      value="65%"
      trend="+5%"
      icon={<Cpu />}
    />
    <MetricCard
      title="Memory"
      value="12 GB / 32 GB"
      trend="-2%"
      icon={<HardDrive />}
    />
    <MetricCard
      title="Queries/Hour"
      value="47"
      trend="+12%"
      icon={<Activity />}
    />
  </div>
  
  {/* Connected Clients */}
  <Card>
    <CardHeader>
      <CardTitle>Connected Clients</CardTitle>
    </CardHeader>
    <CardContent>
      <ClientsTable data={clients} />
    </CardContent>
  </Card>
  
  {/* Document Index Stats */}
  <Card>
    <CardHeader>
      <CardTitle>Document Index</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="space-y-2">
        <StatRow label="Total Documents" value="1,247" />
        <StatRow label="Total Chunks" value="45,892" />
        <StatRow label="Storage Used" value="2.3 GB" />
      </div>
    </CardContent>
  </Card>
  
  {/* Query Volume Chart */}
  <Card>
    <CardHeader>
      <CardTitle>Query Volume (Last 7 Days)</CardTitle>
    </CardHeader>
    <CardContent>
      <LineChart data={queryData} />
    </CardContent>
  </Card>
</div>
```

**Metric Card Component:**
```tsx
<Card>
  <CardHeader className="flex flex-row items-center justify-between pb-2">
    <CardTitle className="text-sm font-medium text-muted-foreground">
      {title}
    </CardTitle>
    <div className="text-muted-foreground">{icon}</div>
  </CardHeader>
  <CardContent>
    <div className="text-3xl font-bold">{value}</div>
    <p className={cn(
      "text-xs",
      trend.startsWith("+") ? "text-green-500" : "text-red-500"
    )}>
      {trend} from last hour
    </p>
  </CardContent>
</Card>
```

**Status Indicators:**
```tsx
const StatusIndicator = ({ status }: { status: "online" | "offline" | "degraded" }) => {
  const colors = {
    online: "bg-green-500",
    offline: "bg-red-500",
    degraded: "bg-yellow-500"
  };
  
  return (
    <div className="flex items-center gap-2">
      <div className={cn("w-2 h-2 rounded-full animate-pulse", colors[status])} />
      <span className="capitalize">{status}</span>
    </div>
  );
};
```

### 3. Document Browser

**File List Layout:**
```tsx
<div className="space-y-4 p-4">
  {/* Search & Filter */}
  <div className="flex gap-2">
    <Input 
      placeholder="Search documents..." 
      icon={<Search />}
      className="flex-1"
    />
    <Select value={filter} onValueChange={setFilter}>
      <SelectTrigger className="w-[180px]">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="all">All Files</SelectItem>
        <SelectItem value="pdf">PDFs Only</SelectItem>
        <SelectItem value="recent">Recent</SelectItem>
      </SelectContent>
    </Select>
  </div>
  
  {/* File Tree */}
  <div className="space-y-2">
    <CollapsibleSection title="NAS Documents" count={1234}>
      {nasFiles.map(file => (
        <FileItem key={file.id} {...file} />
      ))}
    </CollapsibleSection>
    
    <CollapsibleSection title="Local Documents" count={13}>
      {localFiles.map(file => (
        <FileItem key={file.id} {...file} />
      ))}
    </CollapsibleSection>
  </div>
</div>
```

**File Item Component:**
```tsx
<div className="flex items-center justify-between p-3 rounded-md hover:bg-secondary/50 group">
  <div className="flex items-center gap-3">
    <FileIcon type={fileType} className="h-5 w-5" />
    <div>
      <p className="font-medium truncate max-w-[300px]">{filename}</p>
      <p className="text-xs text-muted-foreground">
        {fileSize} • Modified {lastModified}
      </p>
    </div>
  </div>
  
  <div className="flex gap-1 opacity-0 group-hover:opacity-100">
    <Button size="sm" variant="ghost">
      <Eye className="h-4 w-4" />
    </Button>
    <Button size="sm" variant="ghost">
      <Trash2 className="h-4 w-4" />
    </Button>
  </div>
</div>
```

### 4. Settings Panel

**Tabbed Interface:**
```tsx
<Tabs defaultValue="general" className="w-full">
  <TabsList className="grid w-full grid-cols-4">
    <TabsTrigger value="general">General</TabsTrigger>
    <TabsTrigger value="model">Model</TabsTrigger>
    <TabsTrigger value="network">Network</TabsTrigger>
    <TabsTrigger value="privacy">Privacy</TabsTrigger>
  </TabsList>
  
  <TabsContent value="general">
    <GeneralSettings />
  </TabsContent>
  
  <TabsContent value="model">
    <ModelSettings />
  </TabsContent>
  
  <TabsContent value="network">
    <NetworkSettings />
  </TabsContent>
  
  <TabsContent value="privacy">
    <PrivacySettings />
  </TabsContent>
</Tabs>
```

**Form Controls:**
```tsx
{/* Text Input */}
<div className="space-y-2">
  <Label htmlFor="server-url">Server URL</Label>
  <Input 
    id="server-url"
    type="url"
    placeholder="https://llm.local"
    className="h-11" // Touch-friendly height
  />
</div>

{/* Dropdown Select */}
<div className="space-y-2">
  <Label htmlFor="model">Active Model</Label>
  <Select value={model} onValueChange={setModel}>
    <SelectTrigger id="model">
      <SelectValue />
    </SelectTrigger>
    <SelectContent>
      <SelectItem value="mistral-7b">Mistral 7B Instruct</SelectItem>
      <SelectItem value="llama-3-8b">Llama 3.1 (8B)</SelectItem>
      <SelectItem value="phi-3-mini">Phi-3 Mini (3.8B)</SelectItem>
    </SelectContent>
  </Select>
</div>

{/* Slider */}
<div className="space-y-2">
  <div className="flex justify-between">
    <Label htmlFor="temperature">Temperature</Label>
    <span className="text-sm text-muted-foreground">{temperature}</span>
  </div>
  <Slider
    id="temperature"
    min={0}
    max={1}
    step={0.1}
    value={[temperature]}
    onValueChange={([v]) => setTemperature(v)}
  />
</div>

{/* Toggle Switch */}
<div className="flex items-center justify-between">
  <div className="space-y-0.5">
    <Label htmlFor="history">Conversation History</Label>
    <p className="text-sm text-muted-foreground">
      Save chat history locally
    </p>
  </div>
  <Switch
    id="history"
    checked={saveHistory}
    onCheckedChange={setSaveHistory}
  />
</div>

{/* Danger Button */}
<Button variant="destructive" className="w-full">
  <Trash2 className="mr-2 h-4 w-4" />
  Clear All Data
</Button>
```

---

## Interactive Components

### WebSocket Connection Indicator
```tsx
<div className="flex items-center gap-2">
  {status === "connected" && (
    <>
      <div className="relative">
        <div className="w-2 h-2 bg-green-500 rounded-full" />
        <div className="absolute inset-0 w-2 h-2 bg-green-500 rounded-full animate-ping" />
      </div>
      <span className="text-sm text-muted-foreground">Connected to T.A.R.S.</span>
    </>
  )}
  
  {status === "connecting" && (
    <>
      <Loader2 className="w-4 h-4 animate-spin text-yellow-500" />
      <span className="text-sm text-muted-foreground">Connecting...</span>
    </>
  )}
  
  {status === "disconnected" && (
    <>
      <AlertCircle className="w-4 h-4 text-red-500" />
      <span className="text-sm text-muted-foreground">Disconnected</span>
      <Button size="sm" variant="outline" onClick={reconnect}>
        Reconnect
      </Button>
    </>
  )}
</div>
```

### Model Selector Dropdown
```tsx
<Select value={currentModel} onValueChange={handleModelChange}>
  <SelectTrigger className="w-[220px]">
    <Cpu className="mr-2 h-4 w-4" />
    <SelectValue placeholder="Select model" />
  </SelectTrigger>
  <SelectContent>
    <SelectItem value="mistral-7b">
      <div className="flex items-center justify-between w-full">
        <span>Mistral 7B Instruct</span>
        <Badge variant="secondary" className="ml-2">7B</Badge>
      </div>
    </SelectItem>
    <SelectItem value="llama-3-8b">
      <div className="flex items-center justify-between w-full">
        <span>Llama 3.1</span>
        <Badge variant="secondary" className="ml-2">8B</Badge>
      </div>
    </SelectItem>
    <SelectItem value="phi-3-mini">
      <div className="flex items-center justify-between w-full">
        <span>Phi-3 Mini</span>
        <Badge variant="secondary" className="ml-2">3.8B</Badge>
      </div>
    </SelectItem>
  </SelectContent>
</Select>
```

### Document Attachment Interface
```tsx
<div className="flex items-center gap-2">
  <input
    ref={fileInputRef}
    type="file"
    accept=".pdf,.docx,.txt,.md,.csv"
    multiple
    className="hidden"
    onChange={handleFileSelect}
  />
  
  <Button
    variant="ghost"
    size="icon"
    onClick={() => fileInputRef.current?.click()}
    title="Attach files"
  >
    <Paperclip className="h-5 w-5" />
  </Button>
  
  {attachedFiles.length > 0 && (
    <div className="flex gap-1">
      {attachedFiles.map(file => (
        <Badge key={file.name} variant="secondary" className="gap-1">
          <FileIcon type={file.type} className="h-3 w-3" />
          {file.name}
          <button onClick={() => removeFile(file)}>
            <X className="h-3 w-3" />
          </button>
        </Badge>
      ))}
    </div>
  )}
</div>
```

### Loading Skeleton
```tsx
<div className="space-y-3 animate-pulse">
  <div className="h-4 bg-muted rounded w-3/4" />
  <div className="h-4 bg-muted rounded w-1/2" />
  <div className="h-4 bg-muted rounded w-5/6" />
</div>
```

---

## Animation Guidelines

### Micro-interactions
```tsx
// Hover transitions
className="transition-colors duration-150 hover:bg-secondary"

// Button press
className="active:scale-95 transition-transform duration-100"

// Modal entrance
<motion.div
  initial={{ opacity: 0, scale: 0.95 }}
  animate={{ opacity: 1, scale: 1 }}
  exit={{ opacity: 0, scale: 0.95 }}
  transition={{ duration: 0.2 }}
>
  {children}
</motion.div>

// Toast slide-in
<motion.div
  initial={{ y: 50, opacity: 0 }}
  animate={{ y: 0, opacity: 1 }}
  exit={{ y: 50, opacity: 0 }}
  transition={{ type: "spring", stiffness: 300, damping: 30 }}
>
  {toastContent}
</motion.div>
```

### Loading States
```tsx
// Spinner
<Loader2 className="h-4 w-4 animate-spin" />

// Progress bar (indeterminate)
<div className="w-full h-1 bg-muted rounded-full overflow-hidden">
  <div className="h-full bg-primary w-1/3 animate-[shimmer_2s_ease-in-out_infinite]" />
</div>

// Tailwind animation config
module.exports = {
  theme: {
    extend: {
      keyframes: {
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(400%)' }
        }
      }
    }
  }
}
```

### Framer Motion Patterns
```tsx
// Page transition
<motion.div
  initial={{ opacity: 0, x: -20 }}
  animate={{ opacity: 1, x: 0 }}
  exit={{ opacity: 0, x: 20 }}
  transition={{ duration: 0.3 }}
>
  {content}
</motion.div>

// Stagger children (list animation)
<motion.ul
  variants={{
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  }}
  initial="hidden"
  animate="show"
>
  {items.map(item => (
    <motion.li
      key={item.id}
      variants={{
        hidden: { opacity: 0, y: 20 },
        show: { opacity: 1, y: 0 }
      }}
    >
      {item.content}
    </motion.li>
  ))}
</motion.ul>
```

---

## Accessibility Standards

### WCAG 2.1 AA Compliance

**Color Contrast:**
- Normal text: minimum 4.5:1 contrast ratio
- Large text (18pt+): minimum 3:1 contrast ratio
- UI components: minimum 3:1 contrast ratio
- Always test with contrast checker tools

**Keyboard Navigation:**
```tsx
// All interactive elements must be keyboard accessible
<button 
  onClick={handleClick}
  onKeyDown={(e) => e.key === 'Enter' && handleClick()}
>
  Action
</button>

// Skip to main content
<a href="#main-content" className="sr-only focus:not-sr-only">
  Skip to main content
</a>
```

**Focus Indicators:**
```css
/* Clear, visible focus rings */
.focus-visible:focus {
  outline: 2px solid hsl(var(--ring));
  outline-offset: 2px;
}
```

**Screen Reader Support:**
```tsx
// Semantic HTML
<nav aria-label="Main navigation">
  <ul>...</ul>
</nav>

// ARIA labels
<button aria-label="Send message">
  <Send className="h-5 w-5" />
</button>

// Live regions for dynamic content
<div aria-live="polite" aria-atomic="true">
  {statusMessage}
</div>
```

**Text Resize:**
- UI must remain functional at 200% zoom
- Use relative units (rem, em) not fixed pixels
- Test with browser zoom at 200%

### Keyboard Shortcuts
```tsx
const shortcuts = {
  "Ctrl/Cmd + N": "New conversation",
  "Ctrl/Cmd + K": "Focus search",
  "Ctrl/Cmd + ,": "Open settings",
  "Ctrl/Cmd + D": "Toggle dashboard",
  "Ctrl/Cmd + /": "Show keyboard shortcuts",
  "Esc": "Close modal/dropdown",
  "Enter": "Send message",
  "Shift + Enter": "New line in input",
  "Ctrl/Cmd + L": "Clear conversation",
};

// Implementation
useEffect(() => {
  const handleKeyDown = (e: KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      focusSearch();
    }
    // ... other shortcuts
  };
  
  window.addEventListener('keydown', handleKeyDown);
  return () => window.removeEventListener('keydown', handleKeyDown);
}, []);
```

### Touch Targets
- Minimum size: 44x44px (Apple HIG standard)
- Spacing between targets: 8px minimum
- Larger areas on mobile/tablet devices
- Visual feedback on touch (ripple effect)

---

## Performance Optimization

### Code Splitting
```tsx
// Lazy load heavy routes/components
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));
const DocumentBrowser = lazy(() => import('./pages/DocumentBrowser'));

// Usage with Suspense
<Suspense fallback={<LoadingScreen />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/settings" element={<Settings />} />
    <Route path="/documents" element={<DocumentBrowser />} />
  </Routes>
</Suspense>
```

### Virtual Scrolling
```tsx
// For long message lists (>100 items)
import { useVirtualizer } from '@tanstack/react-virtual';

const MessageList = ({ messages }) => {
  const parentRef = useRef(null);
  
  const virtualizer = useVirtualizer({
    count: messages.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 100, // Average message height
    overscan: 5,
  });
  
  return (
    <div ref={parentRef} className="h-full overflow-auto">
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map(virtualRow => (
          <div
            key={virtualRow.index}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              transform: `translateY(${virtualRow.start}px)`,
            }}
          >
            <Message message={messages[virtualRow.index]} />
          </div>
        ))}
      </div>
    </div>
  );
};
```

### Image Optimization
```tsx
// Lazy load images below fold
<img 
  src={imageSrc} 
  alt={altText}
  loading="lazy"
  decoding="async"
/>

// Responsive images
<img
  srcSet={`
    ${image.small} 400w,
    ${image.medium} 800w,
    ${image.large} 1200w
  `}
  sizes="(max-width: 640px) 400px, (max-width: 1024px) 800px, 1200px"
  src={image.medium}
  alt={altText}
/>

// Blur-up placeholder
<div className="relative">
  <img 
    src={blurredPlaceholder} 
    alt=""
    className="absolute inset-0 blur-md"
  />
  <img 
    src={fullImage}
    alt={altText}
    className="relative"
    onLoad={(e) => e.currentTarget.previousElementSibling?.remove()}
  />
</div>
```

### Bundle Size Optimization
```js
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          'vendor-markdown': ['react-markdown', 'remark-gfm'],
          'vendor-charts': ['recharts'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },
});

// Use date-fns instead of moment.js
import { format, parseISO } from 'date-fns';

// Tree-shaking with lodash-es
import debounce from 'lodash-es/debounce';
```

### Bundle Size Targets
- Initial bundle: < 500 KB (gzipped)
- Async chunks: < 200 KB each
- Total transfer: < 2 MB for full app
- Lighthouse Performance score: > 90

---

## Testing Strategy

### Component Testing (Vitest + Testing Library)
```tsx
import { render, screen, userEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MessageBubble } from './MessageBubble';

describe('MessageBubble', () => {
  it('renders user message with correct styling', () => {
    const message = {
      content: 'Hello T.A.R.S.',
      role: 'user',
      timestamp: new Date(),
    };
    
    render(<MessageBubble message={message} />);
    
    expect(screen.getByText('Hello T.A.R.S.')).toBeInTheDocument();
    expect(screen.getByTestId('message-bubble')).toHaveClass('bg-primary');
  });
  
  it('renders assistant message with sources', () => {
    const message = {
      content: 'Based on your documents...',
      role: 'assistant',
      sources: ['document.pdf'],
      timestamp: new Date(),
    };
    
    render(<MessageBubble message={message} />);
    
    expect(screen.getByText(/document\.pdf/)).toBeInTheDocument();
  });
});
```

### Visual Regression Testing
- **Chromatic:** Automated visual testing for Storybook
- **Percy:** Full-page screenshot comparison
- Manual testing on target devices for critical flows

### Accessibility Testing
```tsx
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

it('should have no accessibility violations', async () => {
  const { container } = render(<Dashboard />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

---

## Build Configuration

### Vite Configuration
```ts
// vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@/components': path.resolve(__dirname, './src/components'),
      '@/lib': path.resolve(__dirname, './src/lib'),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-ui': ['@radix-ui/react-*'],
        },
      },
    },
    sourcemap: true,
  },
  server: {
    port: 3000,
    strictPort: true,
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'zustand'],
  },
});
```

### Electron Builder Configuration
```json
{
  "appId": "com.tars.localllm",
  "productName": "T.A.R.S. LocalLLM",
  "directories": {
    "output": "dist-electron",
    "buildResources": "assets"
  },
  "files": [
    "dist/**/*",
    "electron/**/*",
    "package.json"
  ],
  "win": {
    "target": ["nsis", "portable"],
    "icon": "assets/icon.ico",
    "artifactName": "${productName}-${version}-${arch}.${ext}"
  },
  "nsis": {
    "oneClick": false,
    "allowToChangeInstallationDirectory": true,
    "createDesktopShortcut": true,
    "createStartMenuShortcut": true
  },
  "portable": {
    "artifactName": "${productName}-${version}-portable.exe"
  }
}
```

---

## Future Enhancements

### Phase 2 UI Features (Post-MVP)
- [ ] System-aware auto dark/light mode switching
- [ ] Custom theme builder with color picker
- [ ] Conversation branching visualization (tree view)
- [ ] Split-screen markdown preview mode
- [ ] Document annotation and highlighting interface
- [ ] Keyboard-first power user mode

### Phase 3 UI Features
- [ ] Voice input waveform visualization
- [ ] Multi-modal support (image display in chat)
- [ ] Collaborative features (if multi-user added)
- [ ] Plugin marketplace UI
- [ ] Advanced query builder with filters
- [ ] 3D model viewer for supported formats

---

This UI framework document is a living specification and will evolve based on user feedback, performance profiling, and new design trends. Regular design reviews ensure the interface remains modern, accessible, and performant.